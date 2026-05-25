import argparse
import json
import os
import glob
import re
import concurrent.futures
from datetime import datetime

from utils.prompt_utils import extract_json_from_response
from utils.llm_utils import generate_llm_response
from utils.generate_response_utils import safe_model_label


# =============================================================================
# evaluation_pipeline.py — Rubric Generation & Response Evaluation Pipeline
# =============================================================================
#
# Generates per-category rubrics and evaluates tutor responses collected in
# Pipeline 5 using a variable-length 0/1 checklist rubric.
#
# immediate and long_term types are fully independent:
# each produces its own result file and is never merged.
#
# -----------------------------------------------------------------------------
# PIPELINE FLOW
# -----------------------------------------------------------------------------
#
#   [--rubric]
#     Pipeline 3 output (build_scenario)      ─┐
#     Pipeline 4 output (synthesize_dialogue) ─┼─► LLM ──► rubric_{id}_{ts}.json
#     Prompt: generate_rubric.txt             ─┘
#
#   [--evaluate / --evaluate-all]
#     rubric_{id}_{ts}.json                   ─┐
#     Pipeline 4 output (synthesize_dialogue) ─┤
#     Pipeline 5 output (response .txt)       ─┼─► LLM ──► {id}_case_{n}_{type}_{model}_{ts}.json
#     Prompt: evaluate_with_rubric.txt        ─┘
#
#   [--summary-all]
#     evaluation/result/*.json               ──► aggregate_summary_{ts}.json
#                                                (breakdown by model & challenge type)
#
# -----------------------------------------------------------------------------
# USAGE
# -----------------------------------------------------------------------------
#
#   # Generate rubric
#   python evaluation_pipeline.py --rubric --id 0001-01
#   python evaluation_pipeline.py --rubric --id 0001-01 --model openai
#
#   # Evaluate specific response file(s)
#   python evaluation_pipeline.py --evaluate --id 0001-01 \
#       --immediate data/pipeline5/response/0001-01_case_001_immediate_gpt-4o.txt
#   python evaluation_pipeline.py --evaluate --id 0001-01 \
#       --long-term data/pipeline5/response/0001-01_case_001_long_term_gpt-4o.txt
#
#   # Evaluate all responses for a category
#   python evaluation_pipeline.py --evaluate-all --id 0001-01
#   python evaluation_pipeline.py --evaluate-all --id 0001-01 --types immediate
#   python evaluation_pipeline.py --evaluate-all --id 0001-01 --types immediate long_term
#
#   # Parallel workers
#   python evaluation_pipeline.py --evaluate-all --id 0001-01 --workers 4
#
#   # Generate aggregate summary across all categories
#   python evaluation_pipeline.py --summary-all
#
# -----------------------------------------------------------------------------
# OUTPUT DIRECTORY STRUCTURE
# -----------------------------------------------------------------------------
#
#   data/
#   ├── evaluation/
#   │   ├── rubric/
#   │   │   └── rubric_{cat_id}_{ts}.json                        # Generated rubric
#   │   └── result/
#   │       └── {cat_id}_case_{n}_{type}_{model}_{ts}.json       # Evaluation result (per type, independent)
#   └── summaries/
#       ├── evaluation_summary_{cat_id}_{ts}.json                # Per-category summary
#       └── aggregate_summary_{ts}.json                          # Aggregate summary across all categories
#
# -----------------------------------------------------------------------------
# RUBRIC SCORE FORMAT
# -----------------------------------------------------------------------------
#
#   criteria_scores : Variable-length 0/1 checklist (item count varies per category)
#   total_score     : Sum of checklist scores
#   avg_normalized  : total_score / n_criteria  (normalized score for cross-category comparison)
#
# =============================================================================


# ============================================================================================================================
# CONSTANTS
# ============================================================================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_CONFIG_FILE = os.path.join(BASE_DIR, "config", "config_evaluation.json")
_DATA_ROOT_OVERRIDE: str | None = None

def _get_data_root(category_id: str) -> str:
    if _DATA_ROOT_OVERRIDE:
        return _DATA_ROOT_OVERRIDE
    return os.path.join(BASE_DIR, "data")



# ============================================================================================================================
# CONFIG
# ============================================================================================================================
def load_eval_config() -> dict:
    if os.path.exists(EVAL_CONFIG_FILE):
        with open(EVAL_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"  Warning: {EVAL_CONFIG_FILE} not found — using defaults")
    return {}


def get_model_kwargs(cfg: dict, role: str) -> dict:
    entry = cfg.get(role, {})
    return {
        "model_type": entry.get("model_type"),
        "model_name": entry.get("model_name"),
    }


def get_models_to_test(cfg: dict) -> list[str]:
    return [m["name"] for m in cfg.get("models_to_test", [])]


# ============================================================================================================================
# FILE LOADERS
# ============================================================================================================================
def find_latest_file(pattern: str) -> str | None:
    matches = glob.glob(pattern)
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def load_json(path: str, label: str = "") -> dict | None:
    if not os.path.exists(path):
        print(f"  Warning: File not found — {label or path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scenario_data(category_id: str, silent: bool = False) -> dict | None:
    pattern = os.path.join(_get_data_root(category_id), "pipeline3", "build_scenario", f"build_scenario_{category_id}_*.json")
    filepath = find_latest_file(pattern)
    if not filepath:
        if not silent:
            print(f"  Warning: No scenario file found for {category_id}")
        return None
    if not silent:
        print(f"  Loaded scenario       : {os.path.basename(filepath)}")
    return load_json(filepath)


def _rubric_dir(category_id: str) -> str:
    return os.path.join(_get_data_root(category_id), "evaluation", "rubric")

def _result_dir(category_id: str) -> str:
    return os.path.join(_get_data_root(category_id), "evaluation", "result")


def load_rubric(category_id: str) -> tuple[dict, str] | tuple[None, None]:
    pattern = os.path.join(_rubric_dir(category_id), f"rubric_{category_id}_*.json")
    filepath = find_latest_file(pattern)
    if not filepath:
        return None, None
    print(f"  Loaded rubric         : {os.path.basename(filepath)}")
    data = load_json(filepath)
    if data is None:
        return None, None
    return data, filepath


def load_dialogue_for_rubric(category_id: str) -> str:
    dialogue_data = load_dialogue_data(category_id)
    if not dialogue_data:
        print(f"    Dialogue examples     : none found")
        return ""

    cases = []
    for dlg in dialogue_data.get("dialogues", []):
        case_id = dlg.get("case_id", "unknown")
        context = dlg.get("minimal_context", "") or json.dumps(dlg.get("full_context", {}), ensure_ascii=False)
        turns = dlg.get("dialogue", [])
        if not turns:
            continue
        block = f"[{case_id}]\n"
        if context:
            block += f"Scenario: {context}\n"
        block += "Dialogue:\n"
        block += "\n".join(f"{t['speaker'].capitalize()}: {t['message']}" for t in turns)
        cases.append(block)

    print(f"    Dialogue examples     : {len(cases)} cases loaded")
    return "\n\n".join(cases)


def load_dialogue_data(category_id: str) -> dict | None:
    pattern = os.path.join(_get_data_root(category_id), "pipeline4", "synthesize_dialogue", f"synthesize_dialogue_{category_id}_*.json")
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        print(f"  Error: No dialogue file found for {category_id}")
        return None

    # Merge dialogues from all files (multiple scenarios produce separate JSONs)
    merged = None
    all_dialogues = []
    for filepath in all_files:
        data = load_json(filepath)
        if data:
            if merged is None:
                merged = data
            all_dialogues.extend(data.get("dialogues", []))

    if merged is None:
        print(f"  Error: Could not load any dialogue files for {category_id}")
        return None

    merged["dialogues"] = all_dialogues
    print(f"  Loaded dialogue       : {len(all_files)} file(s), {len(all_dialogues)} case(s)")
    return merged


def load_tutor_response(filepath: str) -> str | None:
    candidates = [filepath, f"{filepath}.txt"] if not filepath.endswith(".txt") else [filepath]
    for path in candidates:
        if os.path.exists(path):
            print(f"  Loaded tutor response : {path}")
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
    print(f"  Error: Tutor response file not found — {filepath}")
    return None



# ============================================================================================================================
# STEP 1 — Target student interventions
# ============================================================================================================================
def collect_target_interventions(scenario_data: dict) -> dict:
    result = {
        "effective_immediate": [],
        "effective_long_term": [],
        "ineffective": [],
    }
    for scenario in scenario_data.get("scenarios", []):
        hidden = scenario.get("hidden_strategies", {})
        result["effective_immediate"].extend(hidden.get("effective_immediate", []))
        result["effective_long_term"].extend(hidden.get("effective_long_term", []))
        result["ineffective"].extend(hidden.get("ineffective", []))

    print(f"    Target interventions  : "
          f"{len(result['effective_immediate'])} immediate, "
          f"{len(result['effective_long_term'])} long-term, "
          f"{len(result['ineffective'])} ineffective")
    return result



# ============================================================================================================================
# STEP 2 — Generate rubric via LLM
# ============================================================================================================================
def build_rubric_context(
    scenario_data: dict,
    target_interventions: dict,
    dialogue_examples: str = "",
) -> dict:
    student_persona = scenario_data.get("student_persona", {})
    challenge       = scenario_data.get("challenge_categorization", {})

    has_immediate  = bool(target_interventions.get("effective_immediate"))
    has_long_term  = bool(target_interventions.get("effective_long_term"))
    types = []
    if has_immediate: types.append("immediate")
    if has_long_term: types.append("long_term")
    rubric_types_str = f"Generate rubric for: {', '.join(types)}" if types else "No rubric types available"

    return {
        "STUDENT_DESCRIPTION":       student_persona.get("description", ""),
        "BEHAVIORAL_PATTERNS":       student_persona.get("behavioral_patterns", ""),
        "UNIQUE_TRIGGERS_AND_NEEDS": student_persona.get("unique_triggers_and_needs", ""),
        "APPROACH_EFFECTIVENESS":    student_persona.get("approach_effectiveness", ""),
        "PRIMARY_CHALLENGE":         json.dumps(challenge.get("primary_challenge", {}),  ensure_ascii=False),
        "EFFECTIVE_IMMEDIATE":       json.dumps(target_interventions.get("effective_immediate", []), indent=2, ensure_ascii=False),
        "EFFECTIVE_LONG_TERM":       json.dumps(target_interventions.get("effective_long_term", []),  indent=2, ensure_ascii=False),
        "INEFFECTIVE":               json.dumps(target_interventions.get("ineffective", []),           indent=2, ensure_ascii=False),
        "DIALOGUE_EXAMPLES":         dialogue_examples or "(No dialogue examples available)",
        "RUBRIC_TYPES":              rubric_types_str,
    }


def generate_rubric(category_id: str, cfg: dict, model: str = "auto") -> tuple[dict, str] | tuple[None, None]:
    print("\n" + "=" * 70)
    print("RUBRIC GENERATION")
    print("=" * 70)
    print(f"\nCategory ID : {category_id}\n")

    # Skip if rubric already exists
    existing_rubric, existing_path = load_rubric(category_id)
    if existing_rubric:
        print(f"  [SKIP] Rubric already exists: {os.path.basename(existing_path)}")
        return existing_rubric, existing_path

    rubric_kwargs = get_model_kwargs(cfg, "rubric_model")

    print(">>> Step 1a: Loading target student data...")
    scenario_data = load_scenario_data(category_id)
    if not scenario_data:
        return None, None
    target_interventions = collect_target_interventions(scenario_data)

    print("\n>>> Step 1b: Loading dialogue examples...")
    dialogue_examples = load_dialogue_for_rubric(category_id)

    print("\n>>> Step 2: Generating rubric via LLM...")
    prompt_vars = build_rubric_context(
        scenario_data, target_interventions, dialogue_examples
    )

    raw_response = generate_llm_response("generate_rubric", prompt_vars, **rubric_kwargs)
    if not raw_response:
        print("  Error: No response from LLM")
        return None, None

    try:
        rubric_data = extract_json_from_response(raw_response)
    except Exception as e:
        print(f"  Error: JSON parsing failed — {e}")
        return None, None

    rubric_data["category_id"]  = category_id
    rubric_data["generated_at"] = datetime.now().isoformat()

    rdir = _rubric_dir(category_id)
    os.makedirs(rdir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(rdir, f"rubric_{category_id}_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rubric_data, f, indent=2, ensure_ascii=False)
    print(f"  Rubric saved to: {json_path}")

    return rubric_data, json_path


# ============================================================================================================================
# STEP 3 — Evaluate tutor response using rubric
#   immediate → uses immediate_rubric only
#   long_term → uses long_term_rubric only
# ============================================================================================================================
def parse_response_filename(filename: str):
    name = os.path.basename(filename).replace(".txt", "")
    pattern = r'^(.+?)_case_(\d{3})_(immediate|long_term)_'
    match = re.match(pattern, name)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None


def _extract_model_label(filepath: str) -> str:
    name = os.path.basename(filepath).replace(".txt", "")
    for sep in ("_immediate_", "_long_term_"):
        parts = re.split(sep, name, maxsplit=1)
        if len(parts) == 2:
            return parts[1]
    return name


def load_dialogue_case(dialogue_data: dict, case_num: str) -> dict | None:
    for case in dialogue_data.get("dialogues", []):
        if f"_case_{case_num}" in case.get("case_id", ""):
            return case
    print(f"  Error: case {case_num} not found in dialogue data")
    return None


def _already_evaluated(category_id: str, case_num: str, response_type: str, model_label: str) -> str | None:
    rdir = _result_dir(category_id)
    pattern = os.path.join(rdir, f"{category_id}_case_{case_num}_{response_type}_{model_label}_*.json")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def _evaluate_one(
    category_id:   str,
    case_num:      str,
    response_type: str,
    response_path: str,
    rubric_data:   dict,
    rubric_path:   str,
    dialogue_case: dict,
    eval_kwargs:   dict,
    result_dir:    str | None = None,
) -> dict | None:
    rubric_key     = f"{response_type}_rubric"
    rubric_section = rubric_data.get(rubric_key)
    if not rubric_section:
        reason = "set to null (no evidence for this type)" if rubric_key in rubric_data else "not found in rubric"
        print(f"  Error: '{rubric_key}' {reason} — cannot evaluate {response_type}")
        return None

    tutor_response = load_tutor_response(response_path)
    if not tutor_response:
        return None

    prompt_vars = {
        "RUBRIC":              json.dumps(rubric_section, indent=2, ensure_ascii=False),
        "DIALOGUE":            json.dumps(dialogue_case.get("dialogue", []), indent=2, ensure_ascii=False),
        "FULL_CONTEXT":        json.dumps(dialogue_case.get("full_context", {}), indent=2, ensure_ascii=False),
        "STUDENT_DESCRIPTION": rubric_data.get("student_description", rubric_data.get("category_id", "")),
        "TUTOR_RESPONSE":      tutor_response,
    }

    print(f"  >>> [{response_type}] Calling LLM...")
    raw = generate_llm_response("evaluate_with_rubric", prompt_vars, **eval_kwargs)
    if not raw:
        print(f"  Error: No LLM response ({response_type})")
        return None

    model_label = _extract_model_label(response_path)

    try:
        eval_result = extract_json_from_response(raw)
    except Exception as e:
        print(f"  Error: JSON parsing failed ({response_type}) — {e}")
        return None

    eval_data = {
        "category_id":   category_id,
        "case_num":      case_num,
        "response_type": response_type,
        "evaluated_at":  datetime.now().isoformat(),
        "response_file": os.path.basename(response_path),
        "rubric_file":   os.path.basename(rubric_path),
        "evaluation":    eval_result,
    }

    out_dir = result_dir or _result_dir(category_id)
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem      = f"{category_id}_case_{case_num}_{response_type}_{model_label}_{timestamp}"
    out_path  = os.path.join(out_dir, f"{stem}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    print(f"  >>> [{response_type}] Saved → {out_path}")

    return eval_data


def run_evaluation(
    category_id: str,
    case_num: str,
    cfg: dict,
    immediate_path: str | None = None,
    long_term_path: str | None = None,
    model: str = "auto",
) -> list[dict]:
    print("\n" + "=" * 70)
    print("RUBRIC-BASED EVALUATION")
    print("=" * 70)
    print(f"\nCategory ID : {category_id}  |  Case : {case_num}\n")

    if not immediate_path and not long_term_path:
        print("  Error: At least one of --immediate or --long-term must be provided")
        return []

    eval_kwargs = get_model_kwargs(cfg, "evaluation_model")

    print(">>> Loading rubric...")
    rubric_data, rubric_path = load_rubric(category_id)
    if not rubric_data:
        print("  Rubric not found. Generating...")
        rubric_data, rubric_path = generate_rubric(category_id, cfg=cfg, model=model)
        if not rubric_data or not rubric_path:
            return []

    print(">>> Loading dialogue data...")
    dialogue_data = load_dialogue_data(category_id)
    if not dialogue_data:
        return []
    dialogue_case = load_dialogue_case(dialogue_data, case_num)
    if not dialogue_case:
        return []

    results = []

    rdir = _result_dir(category_id)

    if immediate_path:
        print(f"\n>>> Evaluating IMMEDIATE: {os.path.basename(immediate_path)}")
        result = _evaluate_one(
            category_id, case_num, "immediate",
            immediate_path, rubric_data, rubric_path, dialogue_case, eval_kwargs, rdir,
        )
        if result:
            results.append(result)
            _display_result(result)

    if long_term_path:
        print(f"\n>>> Evaluating LONG_TERM: {os.path.basename(long_term_path)}")
        result = _evaluate_one(
            category_id, case_num, "long_term",
            long_term_path, rubric_data, rubric_path, dialogue_case, eval_kwargs, rdir,
        )
        if result:
            results.append(result)
            _display_result(result)

    return results


# ============================================================================================================================
# EVALUATE ALL
# ============================================================================================================================
def run_evaluate_all(category_id: str, cfg: dict, response_types: list[str],
                     model: str = "auto", workers: int = 1):
    data_root    = _get_data_root(category_id)
    response_dir = os.path.join(data_root, "pipeline5", "response")
    result_dir   = _result_dir(category_id)
    models_to_test = get_models_to_test(cfg)

    rubric_data, rubric_path = load_rubric(category_id)
    if not rubric_data:
        print("\nGenerating rubric first...")
        rubric_data, rubric_path = generate_rubric(category_id, cfg=cfg, model=model)
        if not rubric_data or not rubric_path:
            print("Error: Could not generate rubric")
            return []

    eval_kwargs = get_model_kwargs(cfg, "evaluation_model")

    dialogue_data = load_dialogue_data(category_id)
    if not dialogue_data:
        return []

    jobs: list[tuple[str, str, str, str]] = []

    for response_type in response_types:
        pattern   = os.path.join(response_dir, f"{category_id}_case_*_{response_type}_*.txt")
        all_files = sorted(glob.glob(pattern))

        if models_to_test:
            wanted = {safe_model_label(m) for m in models_to_test}
            all_files = [
                f for f in all_files
                if any(os.path.basename(f).endswith(f"_{m}.txt") for m in wanted)
            ]

        if not all_files:
            print(f"  No {response_type} files found for {category_id} matching models_to_test")
            continue

        for filepath in all_files:
            name  = os.path.basename(filepath)
            # Matches both "0001-01_case_..." and "adhd_0001-01_case_..."
            regex = rf'^(.+?)_case_(\d{{3}})_{re.escape(response_type)}_(.+)\.txt$'
            m     = re.match(regex, name)
            if not m:
                continue
            case_num    = m.group(2)
            model_label = m.group(3)
            jobs.append((case_num, response_type, model_label, filepath))

    if not jobs:
        print(f"No response files found for {category_id}")
        return []

    # Pre-filter: skip already-evaluated, resolve dialogue cases before submitting
    pending_jobs = []
    results = []
    for case_num, response_type, model_label, filepath in sorted(jobs):
        existing = _already_evaluated(category_id, case_num, response_type, model_label)
        if existing:
            print(f"  [SKIP] case_{case_num} | {response_type} | {model_label} — already evaluated")
            d = load_json(existing)
            if d:
                results.append(d)
            continue

        dialogue_case = load_dialogue_case(dialogue_data, case_num)
        if not dialogue_case:
            print(f"  [SKIP] case_{case_num} — dialogue case not found")
            continue

        pending_jobs.append((case_num, response_type, model_label, filepath, dialogue_case))

    print(f"\nFound {len(jobs)} file(s) total, {len(results)} already evaluated, {len(pending_jobs)} to evaluate")

    if not pending_jobs:
        print(f"\n>>> Nothing to evaluate for {category_id}")
        return results

    effective_workers = min(workers, len(pending_jobs))
    if effective_workers > 1:
        print(f"  Running {len(pending_jobs)} evaluation(s) with {effective_workers} workers in parallel...")

        def _run_one(job_tuple):
            case_num, response_type, model_label, filepath, dialogue_case = job_tuple
            print(f"  >>> case_{case_num} | {response_type} | {model_label} — evaluating...", flush=True)
            return _evaluate_one(
                category_id, case_num, response_type,
                filepath, rubric_data, rubric_path, dialogue_case, eval_kwargs, result_dir,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {pool.submit(_run_one, job): job for job in pending_jobs}
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    case_num, response_type, model_label, _, _ = job
                    print(f"  [ERROR] case_{case_num} | {response_type} | {model_label} — {e}")
    else:
        for job in pending_jobs:
            case_num, response_type, model_label, filepath, dialogue_case = job
            print(f"\n>>> case_{case_num} | {response_type} | {model_label}")
            result = _evaluate_one(
                category_id, case_num, response_type,
                filepath, rubric_data, rubric_path, dialogue_case, eval_kwargs, result_dir,
            )
            if result:
                results.append(result)
                _display_result(result)

    print(f"\n>>> Completed {len(results)} evaluation(s) for {category_id}")
    return results



# ============================================================================================================================
# SUMMARY
# ============================================================================================================================
def _extract_score(eval_dict: dict) -> float:
    """Sum of 0/1 checklist scores (variable-length rubric)."""
    total = eval_dict.get("total_score")
    if total is None:
        total = sum(c.get("score", 0) for c in eval_dict.get("criteria_scores", []))
    return float(total)


def _extract_normalized(eval_dict: dict) -> float:
    """Score normalized to [0, 1] = total / n_criteria. Lets us average across categories with different criteria counts."""
    criteria = eval_dict.get("criteria_scores", [])
    n = eval_dict.get("n_criteria") or len(criteria)
    if not n:
        return 0.0
    return _extract_score(eval_dict) / n


def _agg(score_list: list, norm_list: list | None = None) -> dict:
    if not score_list:
        return {}
    out = {
        "count": len(score_list),
        "avg_score": round(sum(score_list) / len(score_list), 3),
    }
    if norm_list:
        out["avg_normalized"] = round(sum(norm_list) / len(norm_list), 3)
    return out


def _build_scores_by_model(scores: dict, norms: dict | None = None) -> dict:
    result = {}
    norms = norms or {}
    for model, type_dict in sorted(scores.items()):
        result[model] = {
            rtype: _agg(slist, norms.get(model, {}).get(rtype))
            for rtype, slist in sorted(type_dict.items())
        }
        all_scores = [s for slist in type_dict.values() for s in slist]
        all_norms  = [n for nlist in norms.get(model, {}).values() for n in nlist]
        result[model]["overall"] = _agg(all_scores, all_norms or None)
    return result


def _get_challenge_type(category_id: str, silent: bool = False) -> str:
    scenario = load_scenario_data(category_id, silent=silent)
    if not scenario:
        return "Unknown"
    return scenario.get("challenge_categorization", {}).get("primary_challenge", {}).get("type", "Unknown")


def generate_evaluation_summary(category_id: str) -> tuple[dict, str] | tuple[None, None]:
    from collections import defaultdict

    rdir = _result_dir(category_id)
    result_files = glob.glob(os.path.join(rdir, f"{category_id}_*.json"))
    if not result_files:
        print(f"  No result files found in {rdir}")
        return None, None

    challenge_type = _get_challenge_type(category_id)

    # model → response_type → list of scores / normalized scores
    scores: dict = defaultdict(lambda: defaultdict(list))
    norms:  dict = defaultdict(lambda: defaultdict(list))

    for fpath in result_files:
        data = load_json(fpath)
        if not data:
            continue
        response_type = data.get("response_type", "")
        model_label   = _extract_model_label(data.get("response_file", ""))
        ev = data.get("evaluation", {})
        scores[model_label][response_type].append(_extract_score(ev))
        norms[model_label][response_type].append(_extract_normalized(ev))

    all_scores_flat = [s for td in scores.values() for slist in td.values() for s in slist]
    summary = {
        "generated_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "category_id":    category_id,
        "challenge_type": challenge_type,
        "stats": {
            "total_evaluated": len(all_scores_flat),
            "scores_by_model": _build_scores_by_model(scores, norms),
        },
    }

    summ_dir = os.path.join(_get_data_root(category_id), "summaries")
    os.makedirs(summ_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(summ_dir, f"evaluation_summary_{category_id}_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Summary saved: {out_path}")
    return summary, out_path


def generate_aggregate_summary() -> tuple[dict, str] | tuple[None, None]:
    """Aggregate evaluation results across ALL categories, broken down by challenge type."""
    from collections import defaultdict

    data_roots = [os.path.join(BASE_DIR, "data")]

    challenge_type_cache: dict[str, str] = {}

    def _cached_challenge_type(cat_id: str) -> str:
        if cat_id not in challenge_type_cache:
            challenge_type_cache[cat_id] = _get_challenge_type(cat_id, silent=True)
        return challenge_type_cache[cat_id]

    # overall: model → response_type → scores
    scores_overall: dict = defaultdict(lambda: defaultdict(list))
    norms_overall:  dict = defaultdict(lambda: defaultdict(list))
    # by_challenge: challenge_type → model → response_type → scores
    scores_by_challenge: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    norms_by_challenge:  dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    total_files = 0
    for data_root in data_roots:
        rdir = os.path.join(data_root, "evaluation", "result")
        if not os.path.exists(rdir):
            continue
        for fpath in sorted(glob.glob(os.path.join(rdir, "*.json"))):
            data = load_json(fpath)
            if not data:
                continue
            cat_id        = data.get("category_id", "")
            response_type = data.get("response_type", "")
            model_label   = _extract_model_label(data.get("response_file", ""))
            ev = data.get("evaluation", {})
            score = _extract_score(ev)
            norm  = _extract_normalized(ev)

            scores_overall[model_label][response_type].append(score)
            norms_overall[model_label][response_type].append(norm)
            ct = _cached_challenge_type(cat_id)
            scores_by_challenge[ct][model_label][response_type].append(score)
            norms_by_challenge[ct][model_label][response_type].append(norm)
            total_files += 1

    if total_files == 0:
        print("  No result files found")
        return None, None

    by_challenge_out = {}
    for ct, model_dict in sorted(scores_by_challenge.items()):
        by_challenge_out[ct] = _build_scores_by_model(model_dict, norms_by_challenge[ct])

    all_scores_flat = [s for td in scores_overall.values() for slist in td.values() for s in slist]
    summary = {
        "generated_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stats": {
            "total_evaluated":    len(all_scores_flat),
            "overall":            _build_scores_by_model(scores_overall, norms_overall),
            "by_challenge_type":  by_challenge_out,
        },
    }

    summ_dir = os.path.join(BASE_DIR, "data", "summaries")
    os.makedirs(summ_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(summ_dir, f"aggregate_summary_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Aggregate summary saved: {out_path}")
    print(f"  Total evaluated: {len(all_scores_flat)} across {len(scores_by_challenge)} challenge type(s)")
    return summary, out_path



# ============================================================================================================================
# DISPLAY
# ============================================================================================================================
def _display_result(eval_data: dict):
    response_type = eval_data.get("response_type", "").upper()
    evaluation    = eval_data.get("evaluation", {})

    print(f"\n{'='*70}")
    print(f"RESULT — {response_type}  |  {eval_data.get('response_file', '')}")
    print(f"{'='*70}")

    criteria_scores = evaluation.get("criteria_scores", [])
    total = 0
    for item in criteria_scores:
        score = item.get("score", 0)
        total += score
        mark  = "✓" if score == 1 else "✗"
        print(f"\n  [{mark}] {item.get('criterion_name', '')}")
        print(f"       {item.get('evidence', '')[:180]}")
    print(f"\n  Score: {total} / {len(criteria_scores)}")

    if evaluation.get("overall_feedback"):
        print(f"\n  Feedback: {evaluation['overall_feedback']}")
    print(f"{'='*70}")



# ============================================================================================================================
# MAIN
# ============================================================================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate personalized rubric and/or evaluate tutor responses"
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rubric",       action="store_true", help="Generate rubric for a category")
    mode.add_argument("--evaluate",     action="store_true", help="Evaluate a specific response file")
    mode.add_argument("--evaluate-all", action="store_true", dest="evaluate_all",
                      help="Evaluate all response files (one job per file, independent)")
    mode.add_argument("--summary-all",  action="store_true", dest="summary_all",
                      help="Aggregate evaluation results across all categories by challenge type")

    parser.add_argument("--id",        default=None, help="Category ID (e.g., 0001-01) — required for all modes except --summary-all")
    parser.add_argument("--case",      default="001",  help="Case number for --evaluate (default: 001)")
    parser.add_argument("--immediate", help="[--evaluate] Path to immediate response .txt file")
    parser.add_argument("--long-term", dest="long_term", help="[--evaluate] Path to long-term response .txt file")
    parser.add_argument("--types",     nargs="+", choices=["immediate", "long_term"],
                        default=["immediate", "long_term"],
                        help="[--evaluate-all] Response types to evaluate (default: both)")
    parser.add_argument("--model",     choices=["auto", "openai", "gemini"], default="auto",
                        help="Fallback LLM engine if config_evaluation.json has no model set")
    parser.add_argument("--config",    default=EVAL_CONFIG_FILE, help="Path to config_evaluation.json")
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument("--workers",        type=int, default=1,
                        help="Max parallel workers for evaluation within a category (default: 1)")
    parser.add_argument("--data-root", dest="data_root", default=None,
                        help="Override data root directory (e.g. experiments/rag)")

    args = parser.parse_args()
    category_id = args.id

    if args.data_root:
        global _DATA_ROOT_OVERRIDE
        _DATA_ROOT_OVERRIDE = os.path.abspath(args.data_root)

    if not args.summary_all and not category_id:
        parser.error("--id is required for this mode")

    if args.config == EVAL_CONFIG_FILE:
        cfg = load_eval_config()
    elif os.path.exists(args.config):
        with open(args.config) as f:
            cfg = json.load(f)
    else:
        cfg = {}

    if args.summary_all:
        generate_aggregate_summary()

    elif args.rubric:
        rubric_data, rubric_path = generate_rubric(category_id, cfg=cfg, model=args.model)
        if not rubric_data:
            print("\nError: Rubric generation failed.")
            return

        if args.verbose:
            print("\n" + "=" * 70)
            print("GENERATED RUBRIC")
            print("=" * 70)
            print(json.dumps(rubric_data, indent=2, ensure_ascii=False))

        print("\n>>> Rubric generation complete!")
        ans = input("\nProceed to evaluate-all now? (y/N): ").strip().lower()
        if ans == "y":
            run_evaluate_all(category_id, cfg=cfg, response_types=["immediate", "long_term"],
                             model=args.model, workers=args.workers)

    elif args.evaluate:
        if not args.immediate and not args.long_term:
            print("Error: --evaluate requires --immediate and/or --long-term")
            return

        case_num = None
        for path in [args.immediate, args.long_term]:
            if path:
                _, parsed_case, _ = parse_response_filename(path)
                if parsed_case:
                    case_num = parsed_case
                    break

        if not case_num:
            case_num = args.case
            print(f"  Note: Could not parse case number from filename, using --case {case_num}")

        if args.case != "001" and case_num != args.case:
            print(f"  Warning: --case {args.case} differs from filename-parsed case {case_num}. Using filename: {case_num}")

        run_evaluation(
            category_id, case_num.zfill(3), cfg,
            immediate_path=args.immediate,
            long_term_path=args.long_term,
            model=args.model,
        )

    elif args.evaluate_all:
        run_evaluate_all(category_id, cfg=cfg, response_types=args.types,
                         model=args.model, workers=args.workers)


if __name__ == "__main__":
    main()