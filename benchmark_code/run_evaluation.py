import argparse
import concurrent.futures
import glob
import os
import re
import subprocess
import sys
from datetime import datetime


# =============================================================================
# run_evaluation.py — Evaluation Batch Runner
# =============================================================================
#
# Automatically scans data/pipeline5/response/ for all response files,
# extracts category IDs from their filenames, and batch-runs
# evaluation_pipeline.py for each category.
#
# No need to specify --id manually for every category.
#
# -----------------------------------------------------------------------------
# PIPELINE FLOW
# -----------------------------------------------------------------------------
#
#   data/pipeline5/response/
#     (auto-collect category IDs from response filenames)
#          │
#          ├─ Step 1: Rubric generation       — skipped if rubric already exists
#          │          evaluation_pipeline.py --rubric --id {cat_id}
#          │
#          ├─ Step 2: Evaluate all responses
#          │          evaluation_pipeline.py --evaluate-all --id {cat_id}
#          │
#          └─ Step 3: Generate aggregate summary
#                     evaluation_pipeline.py --summary-all
#
# -----------------------------------------------------------------------------
# USAGE
# -----------------------------------------------------------------------------
#
#   # Full run (rubric + evaluate + summary)
#   python run_evaluation.py
#
#   # Rubric generation only
#   python run_evaluation.py --rubric-only
#
#   # Evaluation only (rubrics must already exist)
#   python run_evaluation.py --evaluate-only
#
#   # Parallel workers
#   python run_evaluation.py --workers 4
#
#   # Specify response types
#   python run_evaluation.py --types immediate
#   python run_evaluation.py --types immediate long_term
#
# -----------------------------------------------------------------------------
# OUTPUT DIRECTORY STRUCTURE
# -----------------------------------------------------------------------------
#
#   data/
#   ├── evaluation/
#   │   ├── rubric/
#   │   │   └── rubric_{cat_id}_{ts}.json                        # Generated rubrics (skipped if exists)
#   │   └── result/
#   │       └── {cat_id}_case_{n}_{type}_{model}_{ts}.json       # Per-file evaluation results
#   └── summaries/
#       └── aggregate_summary_{ts}.json                          # Final aggregate summary
#
# =============================================================================


# ============================================================================================================================
# HELPERS
# ============================================================================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON   = sys.executable

def find_category_ids(data_root: str) -> list[str]:
    response_dir = os.path.join(data_root, "pipeline5", "response")
    if not os.path.exists(response_dir):
        return []
    pattern = re.compile(r'^(.+?)_case_\d{3}_(immediate|long_term)_.+\.txt$')
    cat_ids: set[str] = set()
    for fname in os.listdir(response_dir):
        m = pattern.match(fname)
        if m:
            cat_ids.add(m.group(1))
    return sorted(cat_ids)


def has_rubric(category_id: str, data_root: str) -> bool:
    pattern = os.path.join(data_root, "evaluation", "rubric", f"rubric_{category_id}_*.json")
    return bool(glob.glob(pattern))


def run_step_captured(label: str, cmd: list[str], stdin_input: str | None = None) -> tuple[bool, str]:
    result = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        input=stdin_input,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    header = f"\n{'─'*70}\n  {label}\n{'─'*70}"
    body   = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        body += f"\n  [ERROR] Step failed (exit code {result.returncode})"
    return result.returncode == 0, header + "\n" + body


Job = tuple[str, list[str], str | None]

def run_parallel(jobs: list[Job], workers: int) -> list[tuple[bool, str]]:
    if not jobs:
        return []
    effective = min(workers, len(jobs))
    print(f"\n  Running {len(jobs)} job(s) with {effective} worker(s) in parallel...")

    results: list[tuple[bool, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=effective) as pool:
        future_to_label = {
            pool.submit(run_step_captured, label, cmd, stdin): label
            for label, cmd, stdin in jobs
        }
        for future in concurrent.futures.as_completed(future_to_label):
            label = future_to_label[future]
            ok, output = future.result()
            print(output, flush=True)
            results.append((ok, label))
    return results


# ============================================================================================================================
# MAIN
# ============================================================================================================================
def main():
    parser = argparse.ArgumentParser(description="Batch evaluation across all categories")
    parser.add_argument("--rubric-only",    action="store_true", dest="rubric_only",   help="Only generate rubrics")
    parser.add_argument("--evaluate-only",  action="store_true", dest="evaluate_only", help="Only run evaluate-all (skip rubric generation)")
    parser.add_argument("--workers",        type=int, default=1, help="Parallel categories (default: 1)")
    parser.add_argument("--types",          nargs="+", choices=["immediate", "long_term"], default=["immediate", "long_term"])
    parser.add_argument("--model",          choices=["auto", "openai", "gemini"], default="auto")
    args = parser.parse_args()

    data_root = os.path.join(BASE_DIR, "data")

    start_time = datetime.now()
    print(f"\n{'='*70}")
    print(f"  EVALUATION BATCH RUN")
    print(f"  Data root      : {data_root}")
    print(f"  Workers        : {args.workers}")
    print(f"  Types          : {', '.join(args.types)}")
    print(f"  Started        : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    category_ids = find_category_ids(data_root)
    if not category_ids:
        print(f"\n[ERROR] No response files found in {data_root}/pipeline5/response/")
        sys.exit(1)

    print(f"\n  Found {len(category_ids)} category/ies")

    errors: list[str] = []

    # ── Step 1: Rubric generation ─────────────────────────────────────────────
    if not args.evaluate_only:
        print(f"\n\n{'#'*70}")
        print(f"  STEP 1 — RUBRIC GENERATION ({len(category_ids)} category/ies)")
        print(f"{'#'*70}")

        jobs: list[Job] = []
        for cat_id in category_ids:
            if has_rubric(cat_id, data_root):
                print(f"  [SKIP] Rubric already exists: {cat_id}")
                continue
            jobs.append((
                f"Rubric: {cat_id}",
                [PYTHON, "evaluation_pipeline.py", "--rubric", "--id", cat_id,
                 "--model", args.model],
                "n\n",
            ))

        print(f"\n  {len(jobs)} rubric(s) to generate, {len(category_ids) - len(jobs)} already exist")
        for ok, label in run_parallel(jobs, args.workers):
            if not ok:
                errors.append(f"[Step 1] Rubric failed: {label}")

    if args.rubric_only:
        _print_summary(errors, category_ids, start_time)
        sys.exit(1 if errors else 0)

    # ── Step 2: Evaluate all ──────────────────────────────────────────────────
    print(f"\n\n{'#'*70}")
    print(f"  STEP 2 — EVALUATE ALL ({len(category_ids)} category/ies)")
    print(f"{'#'*70}")

    jobs = []
    for cat_id in category_ids:
        jobs.append((
            f"Evaluate all: {cat_id}",
            [PYTHON, "evaluation_pipeline.py", "--evaluate-all", "--id", cat_id,
             "--model", args.model,
             "--types"] + args.types,
            None,
        ))

    for ok, label in run_parallel(jobs, args.workers):
        if not ok:
            errors.append(f"[Step 2] Evaluate-all failed: {label}")

    # ── Step 3: Aggregate summary ─────────────────────────────────────────────
    print(f"\n\n{'#'*70}")
    print(f"  STEP 3 — AGGREGATE SUMMARY")
    print(f"{'#'*70}")

    ok, output = run_step_captured(
        "Aggregate summary",
        [PYTHON, "evaluation_pipeline.py", "--summary-all"],
    )
    print(output, flush=True)
    if not ok:
        errors.append("[Step 3] Aggregate summary failed")

    _print_summary(errors, category_ids, start_time)
    sys.exit(1 if errors else 0)


def _print_summary(errors: list[str], category_ids: list[str], start_time: datetime):
    elapsed = datetime.now() - start_time
    print(f"\n\n{'='*70}")
    print(f"  EVALUATION BATCH SUMMARY")
    print(f"  Elapsed    : {str(elapsed).split('.')[0]}")
    print(f"  Categories : {len(category_ids)}")
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors:
            print(f"    x {e}")
    else:
        print(f"\n  All steps completed successfully.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()