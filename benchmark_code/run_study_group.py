import argparse
import concurrent.futures
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime

from utils.generate_response_utils import collect_model_responses, safe_model_label


# =============================================================================
# run_study_group.py — Full-Flow Batch Runner
# =============================================================================
#
# Top-level orchestrator that takes a PDF directory, runs the full data
# construction pipeline (main.py) on each paper, then automatically proceeds
# to rubric generation, response collection, and evaluation
# (evaluation_pipeline.py) — all in one command.
#
# -----------------------------------------------------------------------------
# PIPELINE FLOW
# -----------------------------------------------------------------------------
#
#   data/docs/{study_group}/
#     (scan all *.pdf files in the given directory)
#          │
#          ├─ Step 1: main.py (Pipeline 1–5)                — parallelized per paper
#          │          skipped per paper if full pipeline output already exists
#          │
#          ├─ Step 2: evaluation_pipeline.py --rubric       — per category
#          │          skipped if rubric already exists
#          │
#          ├─ Step 3: generate_missing_responses            — fill in missing model responses
#          │            skipped if all configured model responses already exist
#          │
#          └─ Step 4: evaluation_pipeline.py --evaluate-all — per category
#                     skipped per file if evaluation result already exists
#
# -----------------------------------------------------------------------------
# USAGE
# -----------------------------------------------------------------------------
#
#   # Full run
#   python run_study_group.py data/docs/study_group/
#
#   # Specify LLM engine
#   python run_study_group.py data/docs/study_group/ --model openai
#
#   # Stop after a specific pipeline and generate summary
#   python run_study_group.py data/docs/study_group/ --pipeline1-only
#   python run_study_group.py data/docs/study_group/ --pipeline2-only
#   python run_study_group.py data/docs/study_group/ --pipeline3-only
#   python run_study_group.py data/docs/study_group/ --pipeline4-only
#   python run_study_group.py data/docs/study_group/ --pipeline5-only
#
#   # Start from a later step
#   python run_study_group.py data/docs/study_group/ --from-rubric     # start from Step 2
#   python run_study_group.py data/docs/study_group/ --from-evaluate   # start from Step 4
#
#   # Parallel workers
#   python run_study_group.py data/docs/study_group/ --workers 4
#
#   # Specify response types
#   python run_study_group.py data/docs/study_group/ --types immediate
#
#   # Skip response generation step
#   python run_study_group.py data/docs/study_group/ --skip-response-generation
#
# -----------------------------------------------------------------------------
# OUTPUT DIRECTORY STRUCTURE
# -----------------------------------------------------------------------------
#
#   data/
#   ├── pipeline1/  pipeline2/  pipeline3/  pipeline4/  pipeline5/     # main.py outputs
#   ├── evaluation/
#   │   ├── rubric/
#   │   │   └── rubric_{cat_id}_{ts}.json                              # Generated rubrics
#   │   └── result/
#   │       └── {cat_id}_case_{n}_{type}_{model}_{ts}.json             # Evaluation results
#   └── summaries/
#       ├── pipeline1_summary_{ts}.json                                # Per-pipeline summaries
#       ├── pipeline2_summary_{ts}.json                                #   (generated when
#       ├── pipeline3_summary_{ts}.json                                #    --pipelineN-only
#       ├── pipeline4_summary_{ts}.json                                #    flag is used)
#       ├── pipeline5_summary_{ts}.json
#       └── aggregate_summary_{ts}.json                                # Final aggregate summary
#
# -----------------------------------------------------------------------------
# PIPELINE SUMMARY FLAGS
# -----------------------------------------------------------------------------
#
#   --pipeline1-only  : screen + analyze → pipeline1_summary_{ts}.json
#                       (total PDFs / screen pass-fail / category eligibility)
#   --pipeline2-only  : + extract excerpt → pipeline2_summary_{ts}.json
#                       (excerpt counts, filter breakdown per category)
#   --pipeline3-only  : + build scenario → pipeline3_summary_{ts}.json
#                       (scenario counts, actionable rate, by challenge type)
#   --pipeline4-only  : + synthesize dialogue → pipeline4_summary_{ts}.json
#                       (dialogue counts, turn stats, immediate/long_term split)
#   --pipeline5-only  : + collect responses → pipeline5_summary_{ts}.json
#                       (response counts per model and type)
#
# =============================================================================


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON   = sys.executable


# ============================================================================================================================
# HELPERS
# ============================================================================================================================
def find_pdfs(input_dir: str) -> list[str]:
    all_pdfs = glob.glob(os.path.join(input_dir, "*.pdf"))
    return sorted(all_pdfs)


def extract_paper_id(pdf_path: str) -> str:
    return os.path.splitext(os.path.basename(pdf_path))[0]


def find_category_ids_for_paper(paper_id: str) -> list[str]:
    pattern = os.path.join(
        BASE_DIR, "data", "pipeline3", "build_scenario",
        f"build_scenario_{paper_id}-*_*.json"
    )
    category_ids: set[str] = set()
    for f in glob.glob(pattern):
        name  = os.path.basename(f)
        inner = name.removeprefix("build_scenario_")
        m = re.match(r'^(.+)_\d{8}_\d{6}\.json$', inner)
        cat = m.group(1) if m else None
        if cat:
            category_ids.add(cat)
    return sorted(category_ids)


def has_output(pattern: str) -> bool:
    return bool(glob.glob(pattern))


def load_models_to_test_from_config(config_path: str) -> list[str]:
    if not os.path.exists(config_path):
        return []
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [m["name"] for m in data.get("models_to_test", []) if m.get("name")]
    except Exception:
        return []


def list_dialogue_slots(category_id: str) -> list[tuple[str, str]]:
    dialogue_dir = os.path.join(BASE_DIR, "data", "pipeline4", "dialogue")
    pattern = os.path.join(dialogue_dir, f"{category_id}_case_*_*.txt")
    regex = re.compile(
        rf"^{re.escape(category_id)}_case_(\d{{3}})_(immediate|long_term)\.txt$"
    )
    slots: set[tuple[str, str]] = set()
    for fp in glob.glob(pattern):
        m = regex.match(os.path.basename(fp))
        if m:
            slots.add((m.group(1), m.group(2)))
    return sorted(slots)


def get_eval_coverage(category_id: str, models_to_test: list[str]) -> dict:
    response_dir = os.path.join(BASE_DIR, "data", "pipeline5", "response")
    eval_dir     = os.path.join(BASE_DIR, "data", "evaluation", "result")
    slots        = list_dialogue_slots(category_id)

    status = {
        "category_id": category_id,
        "dialogue_slots": len(slots),
        "models": len(models_to_test),
        "expected_pairs": len(slots) * len(models_to_test),
        "response_exists": 0,
        "evaluated": 0,
        "missing_responses": [],
        "pending_evals": [],
    }

    for case_num, response_type in slots:
        for model_name in models_to_test:
            model_label = safe_model_label(model_name)
            response_name = f"{category_id}_case_{case_num}_{response_type}_{model_label}.txt"
            response_path = os.path.join(response_dir, response_name)

            if os.path.exists(response_path):
                status["response_exists"] += 1
                eval_pattern = os.path.join(
                    eval_dir,
                    f"{category_id}_case_{case_num}_{response_type}_{model_label}_*.json",
                )
                if glob.glob(eval_pattern):
                    status["evaluated"] += 1
                else:
                    status["pending_evals"].append((case_num, response_type, model_label))
            else:
                status["missing_responses"].append((case_num, response_type, model_label))

    return status


def _generate_responses_for_category(
    category_id: str,
    response_models: list[str],
    response_config_path: str,
) -> str:
    cov_before = get_eval_coverage(category_id, response_models)
    missing_before = len(cov_before["missing_responses"])
    if missing_before == 0:
        return f"\n  [{category_id}] missing_response=0\n    [SKIP] All configured model responses already exist."

    try:
        collect_model_responses(category_id, response_config_path)
    except Exception as e:
        return f"\n  [{category_id}] missing_response={missing_before}\n    [ERROR] Response generation failed: {e}"

    cov_after = get_eval_coverage(category_id, response_models)
    missing_after = len(cov_after["missing_responses"])
    generated = max(0, missing_before - missing_after)
    return (
        f"\n  [{category_id}] missing_response={missing_before} → generating...\n"
        f"    done: generated={generated}, now responses={cov_after['response_exists']}, "
        f"still_missing={missing_after}"
    )


def generate_missing_responses(
    category_ids: list[str],
    response_models: list[str],
    response_config_path: str,
    workers: int = 1,
):
    if not response_models:
        print("\n  [WARNING] No models_to_test found in config_response.json")
        print("  Skipping response generation coverage check.")
        return

    print(f"\n\n{'#'*70}")
    print(f"  STEP 3 — GENERATE RESPONSES ({len(category_ids)} category/ies)")
    print(f"{'#'*70}")
    print(f"\n  Models in config_response.json ({len(response_models)}): {', '.join(response_models)}")

    effective_workers = min(workers, len(category_ids))
    print(f"  Running with {effective_workers} worker(s) in parallel...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as pool:
        futures = {
            pool.submit(
                _generate_responses_for_category,
                category_id, response_models, response_config_path,
            ): category_id
            for category_id in category_ids
        }
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)


def run_step(label: str, cmd: list[str], stdin_input: bytes | None = None) -> bool:
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'─'*70}")
    result = subprocess.run(cmd, cwd=BASE_DIR, input=stdin_input)
    if result.returncode != 0:
        print(f"\n  [ERROR] Step failed (exit code {result.returncode})")
        return False
    return True


def run_step_captured(label: str, cmd: list[str], stdin_input: bytes | None = None) -> tuple[bool, str]:
    result = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        input=stdin_input.decode("utf-8") if isinstance(stdin_input, bytes) else stdin_input,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    header = f"\n{'─'*70}\n  {label}\n  {' '.join(cmd)}\n{'─'*70}"
    body   = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        body += f"\n  [ERROR] Step failed (exit code {result.returncode})"
    return result.returncode == 0, header + "\n" + body


Job = tuple[str, list[str], bytes | None]

def run_parallel(jobs: list[Job], workers: int) -> list[tuple[bool, str]]:
    if not jobs:
        return []
    effective_workers = min(workers, len(jobs))
    print(f"\n  Running {len(jobs)} job(s) with {effective_workers} worker(s) in parallel...")

    results: list[tuple[bool, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as pool:
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
# PIPELINE 1 SUMMARY
# ============================================================================================================================
def _load_latest_summary(pipeline_num: int) -> dict | None:
    """Return the stats block of the most recently generated summary for pipeline N, or None."""
    summaries_dir = os.path.join(BASE_DIR, "data", "summaries")
    pattern = os.path.join(summaries_dir, f"pipeline{pipeline_num}_summary_*.json")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not matches:
        return None
    try:
        with open(matches[0], encoding="utf-8") as f:
            data = json.load(f)
        return data.get("stats", {})
    except Exception:
        return None


def generate_pipeline1_summary(paper_ids: list[str], input_dir: str) -> dict:
    """
    Read analyze_paper outputs (which embed screen result + eligible flags) and build a summary.
    Saves to data/summaries/pipeline1_summary_{timestamp}.json and returns the summary dict.
    """
    analyze_dir = os.path.join(BASE_DIR, "data", "pipeline1", "analyze_paper")

    not_processed = []
    screen_rejected = []
    all_cats_filtered = []
    papers_with_categories = []
    n_cats_extracted = 0
    n_cats_eligible = 0

    for paper_id in paper_ids:
        matches = sorted(
            glob.glob(os.path.join(analyze_dir, f"analyze_paper_{paper_id}_*.json")),
            key=os.path.getmtime, reverse=True
        )
        if not matches:
            not_processed.append(paper_id)
            continue

        with open(matches[0]) as f:
            data = json.load(f)

        screen = data.get("screen", {})
        if not screen.get("passed", True):
            screen_rejected.append(paper_id)
            continue

        cats = data.get("category_list", [])
        n_cats_extracted += len(cats)
        eligible = [c for c in cats if c.get("eligible", True)]
        n_cats_eligible += len(eligible)

        if not eligible:
            all_cats_filtered.append(paper_id)
        else:
            papers_with_categories.append(paper_id)

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_dir": input_dir,
        "stats": {
            "total_pdfs": len(paper_ids),
            "not_processed": len(not_processed),
            "screen_rejected": len(screen_rejected),
            "screen_passed": len(paper_ids) - len(not_processed) - len(screen_rejected),
            "all_cats_filtered": len(all_cats_filtered),
            "papers_with_categories": len(papers_with_categories),
            "categories_extracted": n_cats_extracted,
            "categories_eligible": n_cats_eligible,
            "categories_filtered_out": n_cats_extracted - n_cats_eligible,
        },
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(BASE_DIR, "data", "summaries")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"pipeline1_summary_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary, out_path


def generate_pipeline2_summary(paper_ids: list[str]) -> dict:
    """Read pipeline1/analyze_paper + pipeline2 excerpt outputs and build a consolidated summary."""
    analyze_dir = os.path.join(BASE_DIR, "data", "pipeline1", "analyze_paper")
    learner_dir = os.path.join(BASE_DIR, "data", "pipeline2", "extract_excerpt_learner")
    instr_dir   = os.path.join(BASE_DIR, "data", "pipeline2", "extract_excerpt_instruction")

    def _epc_lookup(epc: dict, cat_id: str) -> int:
        if cat_id in epc:
            return epc[cat_id]
        m = re.search(r'\d{4}-\d{2}$', cat_id)
        if m:
            return epc.get(m.group(0), 0)
        return 0

    # Only consider papers that had at least one eligible category in P1
    eligible_papers = []
    paper_p1_data = {}
    for paper_id in paper_ids:
        p1_matches = sorted(
            glob.glob(os.path.join(analyze_dir, f"analyze_paper_{paper_id}_*.json")),
            key=os.path.getmtime, reverse=True
        )
        if not p1_matches:
            continue
        with open(p1_matches[0]) as f:
            p1_data = json.load(f)
        screen = p1_data.get("screen", {})
        if not screen.get("passed", True):
            continue
        cats = p1_data.get("category_list", [])
        if not any(c.get("eligible", True) for c in cats):
            continue
        eligible_papers.append(paper_id)
        paper_p1_data[paper_id] = (p1_matches[0], p1_data)

    not_processed = []
    all_categories = []

    for paper_id in eligible_papers:
        _, p1_data = paper_p1_data[paper_id]
        l_matches = sorted(
            glob.glob(os.path.join(learner_dir, f"extract_excerpt_learner_{paper_id}_*.json")),
            key=os.path.getmtime, reverse=True
        )
        i_matches = sorted(
            glob.glob(os.path.join(instr_dir, f"extract_excerpt_instruction_{paper_id}_*.json")),
            key=os.path.getmtime, reverse=True
        )

        if not l_matches or not i_matches:
            not_processed.append(paper_id)
            continue

        with open(l_matches[0]) as f:
            l_data = json.load(f)
        with open(i_matches[0]) as f:
            i_data = json.load(f)

        learner_epc = l_data.get("extraction_summary", {}).get("excerpts_per_category", {})
        instr_epc   = i_data.get("extraction_summary", {}).get("excerpts_per_category", {})

        for cat in p1_data.get("category_list", []):
            if not cat.get("eligible", True):
                continue
            cat_id  = cat.get("category_id", "")
            l_count = _epc_lookup(learner_epc, cat_id)
            i_count = _epc_lookup(instr_epc, cat_id)
            all_categories.append({
                "category_id":             cat_id,
                "paper_id":                paper_id,
                "pipeline2_passed":        l_count > 0 and i_count > 0,
                "learner_excerpt_count":   l_count,
                "instruction_excerpt_count": i_count,
            })

    passed = [c for c in all_categories if c.get("pipeline2_passed")]

    # Mutually exclusive breakdown of filtered categories (sums to categories_filtered)
    zero_both        = [c for c in all_categories if c.get("learner_excerpt_count", 0) == 0 and c.get("instruction_excerpt_count", 0) == 0]
    zero_learner_only = [c for c in all_categories if c.get("learner_excerpt_count", 0) == 0 and c.get("instruction_excerpt_count", 0) > 0]
    zero_instr_only  = [c for c in all_categories if c.get("learner_excerpt_count", 0) > 0 and c.get("instruction_excerpt_count", 0) == 0]

    import math
    passed_cats     = [c for c in all_categories if c.get("pipeline2_passed")]
    learner_counts  = [c.get("learner_excerpt_count", 0) for c in passed_cats]
    instr_counts    = [c.get("instruction_excerpt_count", 0) for c in passed_cats]
    avg_learner     = sum(learner_counts) / len(learner_counts) if learner_counts else 0
    avg_instruction = sum(instr_counts)   / len(instr_counts)   if instr_counts   else 0
    std_learner     = math.sqrt(sum((x - avg_learner) ** 2 for x in learner_counts) / len(learner_counts))     if learner_counts else 0
    std_instruction = math.sqrt(sum((x - avg_instruction) ** 2 for x in instr_counts) / len(instr_counts)) if instr_counts   else 0
    min_learner     = min(learner_counts) if learner_counts else 0
    max_learner     = max(learner_counts) if learner_counts else 0
    min_instruction = min(instr_counts)   if instr_counts   else 0
    max_instruction = max(instr_counts)   if instr_counts   else 0

    p1_ctx = _load_latest_summary(1) or {}

    papers_with_passing = len(set(c["paper_id"] for c in passed))
    n_filtered = len(zero_both) + len(zero_learner_only) + len(zero_instr_only)
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stats": {
            "input_papers":                  p1_ctx.get("papers_with_categories"),
            "input_categories":              p1_ctx.get("categories_eligible"),
            "papers_not_processed":          len(not_processed),
            "papers_with_passing_categories": papers_with_passing,
            "categories_passed":             len(passed),
            "categories_filtered":           n_filtered,
            "filtered_zero_both":            len(zero_both),
            "filtered_zero_learner_only":    len(zero_learner_only),
            "filtered_zero_instr_only":      len(zero_instr_only),
            "avg_learner_excerpts":          round(avg_learner, 2),
            "std_learner_excerpts":          round(std_learner, 2),
            "min_learner_excerpts":          min_learner,
            "max_learner_excerpts":          max_learner,
            "avg_instruction_excerpts":      round(avg_instruction, 2),
            "std_instruction_excerpts":      round(std_instruction, 2),
            "min_instruction_excerpts":      min_instruction,
            "max_instruction_excerpts":      max_instruction,
        },
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(BASE_DIR, "data", "summaries")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"pipeline2_summary_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary, out_path


def generate_pipeline3_summary(paper_ids: list[str]) -> dict:
    """Read build_scenario outputs and build a consolidated summary."""
    import math
    from collections import Counter, defaultdict

    scenario_dir = os.path.join(BASE_DIR, "data", "pipeline3", "build_scenario")
    categories_processed = []

    for paper_id in paper_ids:
        matches = sorted(
            glob.glob(os.path.join(scenario_dir, f"build_scenario_{paper_id}-*_*.json")),
            key=os.path.getmtime, reverse=True
        )
        seen = {}
        for m in matches:
            fname = os.path.basename(m)
            parts = fname.replace("build_scenario_", "").rsplit("_", 2)
            cat_id = parts[0]
            if cat_id not in seen:
                seen[cat_id] = m

        for cat_id, fpath in seen.items():
            with open(fpath) as f:
                data = json.load(f)
            scenarios = data.get("scenarios", [])
            n = len(scenarios)
            n_actionable = sum(1 for s in scenarios if s.get("is_actionable", True))
            ch = data.get("challenge_categorization", {})
            challenge_type = ch.get("primary_challenge", {}).get("type", "Unknown")
            categories_processed.append({
                "category_id":    cat_id,
                "paper_id":       paper_id,
                "num_scenarios":  n,
                "num_actionable": n_actionable,
                "challenge_type": challenge_type,
            })

    p2_ctx = _load_latest_summary(2) or {}
    input_categories = p2_ctx.get("categories_passed")
    input_papers     = p2_ctx.get("papers_with_passing_categories")

    total_scenarios   = sum(c["num_scenarios"]  for c in categories_processed)
    total_actionable  = sum(c["num_actionable"] for c in categories_processed)
    total_not_actionable = total_scenarios - total_actionable

    n_proc = len(categories_processed)
    n_not_proc = (input_categories - n_proc) if input_categories is not None else None
    cats_with_actionable = sum(1 for c in categories_processed if c["num_actionable"] > 0)
    cats_no_actionable   = n_proc - cats_with_actionable

    # scenario count stats — only categories with actionable scenarios (excluded from P4 otherwise)
    actionable_cats   = [c for c in categories_processed if c["num_actionable"] > 0]
    scenario_counts   = [c["num_scenarios"]  for c in actionable_cats]
    actionable_counts = [c["num_actionable"] for c in actionable_cats]

    def _stats(vals):
        if not vals:
            return {"avg": 0, "std": 0, "min": 0, "max": 0}
        avg = sum(vals) / len(vals)
        std = math.sqrt(sum((x - avg) ** 2 for x in vals) / len(vals))
        return {"avg": round(avg, 2), "std": round(std, 2), "min": min(vals), "max": max(vals)}

    sc_stats  = _stats(scenario_counts)
    act_stats = _stats(actionable_counts)

    scenario_count_dist = dict(sorted(Counter(c["num_scenarios"] for c in actionable_cats).items()))

    type_buckets = defaultdict(lambda: {"categories": 0, "scenarios": 0})
    for c in categories_processed:
        t = c["challenge_type"]
        type_buckets[t]["categories"] += 1
        type_buckets[t]["scenarios"]  += c["num_scenarios"]
    by_challenge_type = {
        t: {"categories": v["categories"], "scenarios": v["scenarios"]}
        for t, v in sorted(type_buckets.items())
    }

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stats": {
            "input_papers":                  input_papers,
            "input_categories":              input_categories,
            "categories_processed":          n_proc,
            "categories_not_processed":      n_not_proc,
            "categories_with_actionable":    cats_with_actionable,
            "categories_no_actionable":      cats_no_actionable,
            "papers_with_scenarios":         None if paper_ids == ["hall"] else len(set(c["paper_id"] for c in categories_processed)),
            "total_scenarios":               total_scenarios,
            "actionable_scenarios":          total_actionable,
            "not_actionable_scenarios":      total_not_actionable,
            "avg_scenarios_per_category":    sc_stats["avg"],
            "std_scenarios_per_category":    sc_stats["std"],
            "min_scenarios_per_category":    sc_stats["min"],
            "max_scenarios_per_category":    sc_stats["max"],
            "avg_actionable_per_category":   act_stats["avg"],
            "std_actionable_per_category":   act_stats["std"],
            "min_actionable_per_category":   act_stats["min"],
            "max_actionable_per_category":   act_stats["max"],
            "scenario_count_distribution":   scenario_count_dist,
        },
        "by_challenge_type": by_challenge_type,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(BASE_DIR, "data", "summaries")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"pipeline3_summary_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary, out_path


def generate_pipeline4_summary(paper_ids: list[str]) -> tuple[dict, str]:
    """Read synthesize_dialogue outputs and build a consolidated summary."""
    scenario_dir = os.path.join(BASE_DIR, "data", "pipeline3", "build_scenario")
    synth_dir    = os.path.join(BASE_DIR, "data", "pipeline4", "synthesize_dialogue")
    dialogue_dir = os.path.join(BASE_DIR, "data", "pipeline4", "dialogue")

    # Collect all actionable categories from P3 (source of truth for input)
    input_cats = {}  # cat_id -> paper_id
    for paper_id in paper_ids:
        for fpath in sorted(
            glob.glob(os.path.join(scenario_dir, f"build_scenario_{paper_id}-*_*.json")),
            key=os.path.getmtime, reverse=True
        ):
            fname = os.path.basename(fpath)
            cat_id = fname.replace("build_scenario_", "").rsplit("_", 2)[0]
            if cat_id in input_cats:
                continue
            with open(fpath) as f:
                data = json.load(f)
            if any(s.get("is_actionable", True) for s in data.get("scenarios", [])):
                input_cats[cat_id] = paper_id

    # Check which input categories got synthesize_dialogue output
    import math
    processed_cats = set()
    dialogues_per_cat = []
    turns_per_dialogue = []

    for cat_id in input_cats:
        matches = sorted(
            glob.glob(os.path.join(synth_dir, f"synthesize_dialogue_{cat_id}_*.json")),
            key=os.path.getmtime, reverse=True
        )
        if not matches:
            continue
        processed_cats.add(cat_id)
        cat_dialogues = 0
        for fpath in matches:
            with open(fpath) as f:
                data = json.load(f)
            for dlg in data.get("dialogues", []):
                cat_dialogues += 1
                turns_per_dialogue.append(len(dlg.get("dialogue", [])))
        dialogues_per_cat.append(cat_dialogues)

    papers_with_dialogue = len({input_cats[c] for c in processed_cats})
    papers_not_processed = len({pid for pid in input_cats.values()}) - papers_with_dialogue

    txt_files     = glob.glob(os.path.join(dialogue_dir, "*.txt"))
    immediate_txt = [f for f in txt_files if "_immediate.txt" in f]
    long_term_txt = [f for f in txt_files if "_long_term.txt" in f]

    def _stats(vals):
        if not vals:
            return {"avg": 0, "std": 0, "min": 0, "max": 0}
        avg = sum(vals) / len(vals)
        std = math.sqrt(sum((x - avg) ** 2 for x in vals) / len(vals))
        return {"avg": round(avg, 2), "std": round(std, 2), "min": min(vals), "max": max(vals)}

    total_dialogues = sum(dialogues_per_cat)
    p3_ctx = _load_latest_summary(3) or {}

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stats": {
            "input_categories":           p3_ctx.get("categories_with_actionable"),
            "input_actionable_scenarios": p3_ctx.get("actionable_scenarios"),
            "input_papers":               len(set(input_cats.values())),
            "categories_processed":       len(processed_cats),
            "categories_not_processed":   len(input_cats) - len(processed_cats),
            "papers_with_dialogue":       papers_with_dialogue,
            "papers_not_processed":       papers_not_processed,
            "total_dialogues":            total_dialogues,
            "immediate_txt":              len(immediate_txt),
            "long_term_txt":              len(long_term_txt),
            "total_dialogue_txt":         len(txt_files),
            "avg_dialogues_per_category": _stats(dialogues_per_cat)["avg"],
            "std_dialogues_per_category": _stats(dialogues_per_cat)["std"],
            "min_dialogues_per_category": _stats(dialogues_per_cat)["min"],
            "max_dialogues_per_category": _stats(dialogues_per_cat)["max"],
            "avg_turns_per_dialogue":     _stats(turns_per_dialogue)["avg"],
            "std_turns_per_dialogue":     _stats(turns_per_dialogue)["std"],
            "min_turns_per_dialogue":     _stats(turns_per_dialogue)["min"],
            "max_turns_per_dialogue":     _stats(turns_per_dialogue)["max"],
        },
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(BASE_DIR, "data", "summaries")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"pipeline4_summary_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary, out_path


def generate_pipeline5_summary(paper_ids: list[str]) -> tuple[dict, str]:
    """Read response files and build a consolidated summary."""
    response_dir = os.path.join(BASE_DIR, "data", "pipeline5", "response")
    all_files = glob.glob(os.path.join(response_dir, "*.txt"))

    # Filter to only files matching known papers
    paper_set = set(paper_ids)
    relevant = []
    for f in all_files:
        fname = os.path.basename(f)
        paper_id = fname.split("-")[0]
        if paper_id in paper_set:
            relevant.append(fname)

    from collections import Counter
    model_counts = Counter()
    type_counts  = Counter()
    for fname in relevant:
        m = re.search(r'_(immediate|long_term)_(.+?)\.txt$', fname)
        if m:
            type_counts[m.group(1)] += 1
            model_counts[m.group(2)] += 1

    p4_ctx = _load_latest_summary(4) or {}

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stats": {
            "input_dialogue_txt":  p4_ctx.get("total_dialogue_txt"),
            "input_immediate_txt": p4_ctx.get("immediate_txt"),
            "input_long_term_txt": p4_ctx.get("long_term_txt"),
            "total_responses":    len(relevant),
            "immediate":          type_counts.get("immediate", 0),
            "long_term":          type_counts.get("long_term", 0),
            "models":             len(model_counts),
            "responses_by_model": dict(model_counts.most_common()),
        },
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(BASE_DIR, "data", "summaries")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"pipeline5_summary_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary, out_path


def print_pipeline1_summary(summary: dict, out_path: str):
    s = summary["stats"]
    print(f"\n\n{'='*70}")
    print(f"  PIPELINE 1 SUMMARY")
    print(f"  Input dir             : {summary['input_dir']}")
    print(f"{'─'*70}")
    print(f"  Total PDFs            : {s['total_pdfs']}")
    print(f"  Not processed         : {s['not_processed']}")
    print(f"  Screen rejected       : {s['screen_rejected']}")
    print(f"  Screen passed         : {s['screen_passed']}")
    print(f"{'─'*70}")
    print(f"  All cats filtered out : {s['all_cats_filtered']}")
    print(f"  Papers with categories: {s['papers_with_categories']}")
    print(f"{'─'*70}")
    print(f"  Categories extracted  : {s['categories_extracted']}")
    print(f"  Categories eligible   : {s['categories_eligible']}")
    print(f"  Categories filtered   : {s['categories_filtered_out']}")
    print(f"\n  Summary saved: {out_path}")
    print(f"{'='*70}\n")



# ============================================================================================================================
# MAIN
# ============================================================================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run the full pipeline for a directory of PDFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_dir",
        help="Path to directory containing PDFs (e.g. papers/case_study/)",
    )
    parser.add_argument(
        "--model",
        choices=["auto", "openai", "gemini"],
        default="auto",
        help="LLM engine for all steps (default: auto)",
    )
    parser.add_argument(
        "--pipeline1-only",
        action="store_true",
        dest="pipeline1_only",
        help="Run only screen + Pipeline 1 (analyze), then generate summary",
    )
    parser.add_argument(
        "--pipeline2-only",
        action="store_true",
        dest="pipeline2_only",
        help="Run Pipelines 1-2 (screen + analyze + extract excerpt), then generate summary",
    )
    parser.add_argument(
        "--pipeline3-only",
        action="store_true",
        dest="pipeline3_only",
        help="Run Pipelines 1-3 (through build scenario), then generate summary",
    )
    parser.add_argument(
        "--pipeline4-only",
        action="store_true",
        dest="pipeline4_only",
        help="Generate Pipeline 4 summary (synthesize_dialogue outputs)",
    )
    parser.add_argument(
        "--pipeline5-only",
        action="store_true",
        dest="pipeline5_only",
        help="Generate Pipeline 5 summary (response outputs)",
    )
    parser.add_argument(
        "--from-rubric",
        action="store_true",
        dest="from_rubric",
        help="Skip paper processing; start from rubric generation",
    )
    parser.add_argument(
        "--from-evaluate",
        action="store_true",
        dest="from_evaluate",
        help="Skip everything except evaluate-all (rubric must already exist)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Max parallel workers (default: 1)",
    )
    parser.add_argument(
        "--response-config",
        default=os.path.join(BASE_DIR, "config", "config_response.json"),
        help="Path to config_response.json",
    )
    parser.add_argument(
        "--skip-response-generation",
        action="store_true",
        help="Skip response generation step",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["immediate", "long_term"],
        default=["immediate", "long_term"],
        help="Response types to evaluate (default: both)",
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir.rstrip("/\\"))

    start_time = datetime.now()
    print(f"\n{'='*70}")
    print(f"  PIPELINE BATCH RUN")
    print(f"  Directory : {input_dir}")
    print(f"  Model     : {args.model}")
    if args.pipeline1_only:
        print(f"  Mode      : pipeline1-only (screen + analyze)")
    print(f"  Started   : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    if not os.path.isdir(input_dir):
        print(f"\n[ERROR] Directory not found: {input_dir}")
        sys.exit(1)

    pdfs = find_pdfs(input_dir)
    if not pdfs:
        print(f"\n[ERROR] No PDFs found in {input_dir}")
        sys.exit(1)

    print(f"\n  Found {len(pdfs)} PDF(s)")

    errors: list[str] = []

    # ── Step 1: Screen + Analyze Paper ───────────────────────────────────────
    if not args.from_rubric and not args.from_evaluate:
        mode_label = "SCREEN + PIPELINE 1 (ANALYZE)" if args.pipeline1_only else "SCREEN + FULL PIPELINE"
        print(f"\n\n{'#'*70}")
        print(f"  STEP 1 — {mode_label} ({len(pdfs)} paper(s))")
        print(f"{'#'*70}")

        jobs: list[Job] = []
        for pdf_path in pdfs:
            paper_id = extract_paper_id(pdf_path)

            # Skip if pipeline1-only and screen result already exists
            if args.pipeline1_only:
                screen_pattern = os.path.join(
                    BASE_DIR, "data", "pipeline1", "screen_paper",
                    f"screen_paper_{paper_id}_*.json"
                )
                analyze_pattern = os.path.join(
                    BASE_DIR, "data", "pipeline1", "analyze_paper",
                    f"analyze_paper_{paper_id}_*.json"
                )
                # Only skip if both screen AND analyze outputs exist (or screen rejected)
                if has_output(screen_pattern):
                    screen_matches = sorted(glob.glob(screen_pattern), key=os.path.getmtime, reverse=True)
                    with open(screen_matches[0]) as f:
                        sr = json.load(f)
                    if not sr.get("passed", False) or has_output(analyze_pattern):
                        print(f"  [SKIP] {paper_id} — output already exists")
                        continue
            elif args.pipeline4_only:
                # Skip if no P3 output (paper not eligible or not yet processed)
                p3_pattern = os.path.join(
                    BASE_DIR, "data", "pipeline3", "build_scenario",
                    f"build_scenario_{paper_id}-*_*.json"
                )
                if not has_output(p3_pattern):
                    print(f"  [SKIP] {paper_id} — no P3 output")
                    continue
                # Let main.py handle scenario-level skip for partial P4 outputs
            elif args.pipeline5_only:
                # Skip if no P4 output
                p4_pattern = os.path.join(
                    BASE_DIR, "data", "pipeline4", "synthesize_dialogue",
                    f"synthesize_dialogue_{paper_id}-*_*.json"
                )
                if not has_output(p4_pattern):
                    print(f"  [SKIP] {paper_id} — no P4 output")
                    continue
                # Let main.py handle dialogue-level skip for partial P5 outputs
            else:
                synth_pattern = os.path.join(
                    BASE_DIR, "data", "pipeline4", "synthesize_dialogue",
                    f"synthesize_dialogue_{paper_id}-*_*.json"
                )
                if has_output(synth_pattern):
                    print(f"  [SKIP] {paper_id} — full pipeline output already exists")
                    continue

            cmd = [PYTHON, "main.py", pdf_path, "--model", args.model]
            if args.pipeline1_only:
                cmd.append("--pipeline1-only")
            elif args.pipeline2_only:
                cmd.append("--pipeline2-only")
            elif args.pipeline3_only:
                cmd.append("--pipeline3-only")
            elif args.pipeline4_only:
                cmd.append("--pipeline4-only")
            elif args.pipeline5_only:
                cmd.append("--pipeline5-only")

            jobs.append((f"Pipeline: {paper_id}", cmd, None))

        for ok, label in run_parallel(jobs, args.workers):
            if not ok:
                errors.append(f"[Step 1] Failed: {label}")

    # ── Pipeline1-only: generate summary and exit ─────────────────────────────
    if args.pipeline1_only:
        paper_ids = [extract_paper_id(p) for p in pdfs]
        summary, out_path = generate_pipeline1_summary(paper_ids, input_dir)
        print_pipeline1_summary(summary, out_path)
        sys.exit(1 if errors else 0)

    # ── Pipeline2-only: generate summary and exit ─────────────────────────────
    if args.pipeline2_only:
        paper_ids = [extract_paper_id(p) for p in pdfs]
        summary, out_path = generate_pipeline2_summary(paper_ids)
        s = summary["stats"]
        print(f"\n{'='*70}")
        print(f"  PIPELINE 2 SUMMARY")
        print(f"{'─'*70}")
        print(f"  [P1 → P2] Input papers          : {s['input_papers']}")
        print(f"  [P1 → P2] Input categories      : {s['input_categories']}")
        print(f"{'─'*70}")
        print(f"  Papers not processed      : {s['papers_not_processed']}")
        print(f"  Categories passed         : {s['categories_passed']}")
        print(f"  Categories filtered       : {s['categories_filtered']}")
        print(f"    zero both               :   {s['filtered_zero_both']}")
        print(f"    zero learner only       :   {s['filtered_zero_learner_only']}")
        print(f"    zero instr only         :   {s['filtered_zero_instr_only']}")
        print(f"  Avg learner excerpts      : {s['avg_learner_excerpts']} (std {s['std_learner_excerpts']}, min {s['min_learner_excerpts']}, max {s['max_learner_excerpts']})")
        print(f"  Avg instruction excerpts  : {s['avg_instruction_excerpts']} (std {s['std_instruction_excerpts']}, min {s['min_instruction_excerpts']}, max {s['max_instruction_excerpts']})")
        print(f"\n  Summary saved: {out_path}")
        print(f"{'='*70}\n")
        sys.exit(1 if errors else 0)

    # ── Pipeline3-only: generate summary and exit ─────────────────────────────
    if args.pipeline3_only:
        paper_ids = [extract_paper_id(p) for p in pdfs]
        summary, out_path = generate_pipeline3_summary(paper_ids)
        s = summary["stats"]
        print(f"\n{'='*70}")
        print(f"  PIPELINE 3 SUMMARY")
        print(f"{'─'*70}")
        print(f"  [P2 → P3] Input papers          : {s['input_papers']}")
        print(f"  [P2 → P3] Input categories      : {s['input_categories']}")
        print(f"{'─'*70}")
        print(f"  Categories processed      : {s['categories_processed']}")
        print(f"  Categories not processed  : {s['categories_not_processed']}")
        print(f"  Categories w/ actionable  : {s['categories_with_actionable']}")
        print(f"  Categories no actionable  : {s['categories_no_actionable']}")
        print(f"  Papers with scenarios     : {s['papers_with_scenarios']}")
        print(f"  Total scenarios           : {s['total_scenarios']}")
        print(f"  Actionable scenarios      : {s['actionable_scenarios']}")
        print(f"  Not actionable            : {s['not_actionable_scenarios']}")
        print(f"  Avg scenarios/category    : {s['avg_scenarios_per_category']} (std {s['std_scenarios_per_category']}, min {s['min_scenarios_per_category']}, max {s['max_scenarios_per_category']})")
        print(f"  Avg actionable/category   : {s['avg_actionable_per_category']} (std {s['std_actionable_per_category']}, min {s['min_actionable_per_category']}, max {s['max_actionable_per_category']})")
        print(f"  Scenario count dist       : {s['scenario_count_distribution']}")
        print(f"\n{'─'*70}")
        print(f"  {'Challenge Type':<40} {'Categories':>12} {'Scenarios':>10}")
        print(f"{'─'*70}")
        for ctype, v in summary.get("by_challenge_type", {}).items():
            print(f"  {ctype:<40} {v['categories']:>12} {v['scenarios']:>10}")
        print(f"\n  Summary saved: {out_path}")
        print(f"{'='*70}\n")
        sys.exit(1 if errors else 0)

    # ── Pipeline4-only: generate summary and exit ─────────────────────────────
    if args.pipeline4_only:
        paper_ids = [extract_paper_id(p) for p in pdfs]
        summary, out_path = generate_pipeline4_summary(paper_ids)
        s = summary["stats"]
        print(f"\n{'='*70}")
        print(f"  PIPELINE 4 SUMMARY")
        print(f"{'─'*70}")
        print(f"  [P3 → P4] Input categories      : {s['input_categories']}")
        print(f"  [P3 → P4] Input actionable scen : {s['input_actionable_scenarios']}")
        print(f"  [P3 → P4] Input papers          : {s['input_papers']}")
        print(f"{'─'*70}")
        print(f"  Categories processed      : {s['categories_processed']}")
        print(f"  Categories not processed  : {s['categories_not_processed']}")
        print(f"  Papers with dialogue      : {s['papers_with_dialogue']}")
        print(f"  Papers not processed      : {s['papers_not_processed']}")
        print(f"{'─'*70}")
        print(f"  Total dialogues           : {s['total_dialogues']}")
        print(f"  Immediate / Long-term txt : {s['immediate_txt']} / {s['long_term_txt']}")
        print(f"  Total dialogue txt files  : {s['total_dialogue_txt']}")
        print(f"  Dialogues/category        : avg {s['avg_dialogues_per_category']} (std {s['std_dialogues_per_category']}, min {s['min_dialogues_per_category']}, max {s['max_dialogues_per_category']})")
        print(f"  Turns/dialogue            : avg {s['avg_turns_per_dialogue']} (std {s['std_turns_per_dialogue']}, min {s['min_turns_per_dialogue']}, max {s['max_turns_per_dialogue']})")
        print(f"\n  Summary saved: {out_path}")
        print(f"{'='*70}\n")
        sys.exit(1 if errors else 0)

    # ── Pipeline5-only: generate summary and exit ─────────────────────────────
    if args.pipeline5_only:
        paper_ids = [extract_paper_id(p) for p in pdfs]
        summary, out_path = generate_pipeline5_summary(paper_ids)
        s = summary["stats"]
        print(f"\n{'='*70}")
        print(f"  PIPELINE 5 SUMMARY")
        print(f"{'─'*70}")
        print(f"  [P4 → P5] Input dialogue txt    : {s['input_dialogue_txt']}")
        print(f"  [P4 → P5] Immediate / Long-term : {s['input_immediate_txt']} / {s['input_long_term_txt']}")
        print(f"{'─'*70}")
        print(f"  Total responses           : {s['total_responses']}")
        print(f"  Immediate / Long-term     : {s['immediate']} / {s['long_term']}")
        print(f"  Models                    : {s['models']}")
        for model, cnt in s['responses_by_model'].items():
            print(f"    {cnt:5d}  {model}")
        print(f"\n  Summary saved: {out_path}")
        print(f"{'='*70}\n")
        sys.exit(1 if errors else 0)

    # ── Discover category IDs ─────────────────────────────────────────────────
    all_category_ids: list[str] = []
    for pdf_path in pdfs:
        paper_id = extract_paper_id(pdf_path)
        all_category_ids.extend(find_category_ids_for_paper(paper_id))
    all_category_ids = sorted(set(all_category_ids))

    if not all_category_ids:
        print(f"\n[WARNING] No category IDs found in data/pipeline3/build_scenario/")
        print("  If Step 1 ran, check data/logs/ for errors in scenario generation.")
        _print_summary(errors, pdfs, [], start_time)
        sys.exit(1 if errors else 0)

    print(f"\n\n  Category IDs ({len(all_category_ids)}): {', '.join(all_category_ids)}")

    # ── Step 2: Rubric generation ──────────────────────────────────────────────
    if not args.from_evaluate:
        print(f"\n\n{'#'*70}")
        print(f"  STEP 2 — RUBRIC GENERATION ({len(all_category_ids)} category/ies)")
        print(f"{'#'*70}")
        jobs = []
        for category_id in all_category_ids:
            rubric_pattern = os.path.join(BASE_DIR, "data", "evaluation", "rubric", f"rubric_{category_id}_*.json")
            if has_output(rubric_pattern):
                print(f"\n  [SKIP] Rubric already exists for '{category_id}'")
                continue
            jobs.append((
                f"Generate rubric: {category_id}",
                [PYTHON, "evaluation_pipeline.py", "--rubric", "--id", category_id, "--model", args.model],
                b"n\n",
            ))
        for ok, label in run_parallel(jobs, args.workers):
            if not ok:
                errors.append(f"[Step 2] Rubric generation failed: {label}")

    # ── Step 3: Generate missing responses ──────────────────────────────────
    if not args.skip_response_generation:
        response_models = load_models_to_test_from_config(args.response_config)
        generate_missing_responses(
            category_ids=all_category_ids,
            response_models=response_models,
            response_config_path=args.response_config,
            workers=args.workers,
        )

    # ── Step 4: Evaluate all ───────────────────────────────────────────────────
    print(f"\n\n{'#'*70}")
    print(f"  STEP 4 — EVALUATE ALL ({len(all_category_ids)} category/ies)")
    print(f"{'#'*70}")

    eval_cfg_path   = os.path.join(BASE_DIR, "config", "config_evaluation.json")
    models_to_check = load_models_to_test_from_config(eval_cfg_path)
    if models_to_check:
        print(f"\n  Models in config_evaluation.json ({len(models_to_check)}): {', '.join(models_to_check)}")
    else:
        print("\n  [WARNING] No models_to_test found in config_evaluation.json")

    jobs = []
    coverage_rows = []

    for category_id in all_category_ids:
        if models_to_check:
            cov = get_eval_coverage(category_id, models_to_check)
            coverage_rows.append(cov)

            print(
                f"\n  [{category_id}] slots={cov['dialogue_slots']}, models={cov['models']}, "
                f"expected={cov['expected_pairs']}, responses={cov['response_exists']}, "
                f"evaluated={cov['evaluated']}, pending_eval={len(cov['pending_evals'])}, "
                f"missing_response={len(cov['missing_responses'])}"
            )

            if cov["expected_pairs"] == 0:
                print("    [SKIP] No dialogue slots found for this category.")
                continue
            if not cov["pending_evals"]:
                print("    [SKIP] No pending evaluations.")
                continue

        jobs.append((
            f"Evaluate all: {category_id}",
            [PYTHON, "evaluation_pipeline.py", "--evaluate-all", "--id", category_id,
             "--model", args.model, "--workers", str(args.workers),
             "--types"] + args.types,
            None,
        ))

    if coverage_rows:
        total_expected = sum(r["expected_pairs"] for r in coverage_rows)
        total_resp     = sum(r["response_exists"] for r in coverage_rows)
        total_eval     = sum(r["evaluated"] for r in coverage_rows)
        total_pending  = sum(len(r["pending_evals"]) for r in coverage_rows)
        total_missing  = sum(len(r["missing_responses"]) for r in coverage_rows)
        print(f"\n  Coverage summary (all categories):")
        print(f"    Expected pairs : {total_expected}")
        print(f"    Responses found: {total_resp}")
        print(f"    Evaluated      : {total_eval}")
        print(f"    Pending eval   : {total_pending}")
        print(f"    Missing resp   : {total_missing}")

    for ok, label in run_parallel(jobs, args.workers):
        if not ok:
            errors.append(f"[Step 4] Evaluate-all failed: {label}")

    _print_summary(errors, pdfs, all_category_ids, start_time)
    sys.exit(1 if errors else 0)


def _print_summary(
    errors: list[str],
    pdfs: list[str],
    category_ids: list[str],
    start_time: datetime,
):
    elapsed = datetime.now() - start_time
    print(f"\n\n{'='*70}")
    print(f"  PIPELINE BATCH SUMMARY")
    print(f"  Elapsed   : {str(elapsed).split('.')[0]}")
    print(f"  Papers    : {len(pdfs)}")
    print(f"  Categories: {len(category_ids)}")
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors:
            print(f"    x {e}")
    else:
        print(f"\n  All steps completed successfully.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()