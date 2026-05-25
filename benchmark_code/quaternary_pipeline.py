import argparse
import concurrent.futures
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime

from utils.llm_utils import generate_llm_response
from utils.prompt_utils import extract_json_from_response as extract_json


# =============================================================================
# quaternary_pipeline.py — MCQ (4-Option) Generation & Evaluation Pipeline
# =============================================================================
#
# Builds, runs, and summarizes 4-option multiple-choice questions for
# evaluating LLM tutors. Each question uses one effective intervention from
# the target scenario as the correct answer, and three effective interventions
# from other scenarios as distractors.
#
# Distractors are selected from scenarios with similar task context but
# different student behaviors and intervention types. Embedding-based scoring
# penalizes near-duplicate candidates; an LLM judge picks the final 3 and
# flags any forced-fill slots.
#
# -----------------------------------------------------------------------------
# PIPELINE FLOW
# -----------------------------------------------------------------------------
#
#   [--build]  Question Construction
#     Pipeline 3 output (build_scenario)  ──► Embed task_context / behavior /
#     Pipeline 4 output (dialogue .txt)       intervention per scenario
#                                         ──► Top-K nearest scenarios
#                                             (excluding same-paper / same-category)
#                                         ──► LLM judge selects 3 best distractors
#                                             (flags forced picks)
#                                         ──► LLM adapter surface-aligns 4 options
#                                         ──► Randomize ABCD assignment
#                                         ──► experiments/mcq/quaternary/questions/{question_id}.json
#
#   [--run]  Model Evaluation
#     questions/{question_id}.json        ──► Feed to all models in config_response.json
#     Prompt: mcq_evaluate.txt            ──► experiments/mcq/quaternary/results/{question_id}_{model}_{ts}.json
#
#   [--summary]  Accuracy Aggregation
#     results/*.json                      ──► experiments/mcq/quaternary/summary/quaternary_summary_{ts}.json
#                                             (per-model accuracy, by challenge type & response type)
#
# -----------------------------------------------------------------------------
# DISTRACTOR SELECTION SCORING
# -----------------------------------------------------------------------------
#
#   score = ctx_sim × (1 - behav_sim) × (1 - max_interv_sim)
#
#   ctx_sim        : cosine similarity of task context embeddings
#   behav_sim      : cosine similarity of behavior embeddings (higher = more similar behavior, penalized)
#   max_interv_sim : max cosine similarity between candidate intervention and ANY of
#                    the target's effective interventions (penalizes near-duplicates)
#
#   Each scenario is represented by its single best-scoring effective intervention.
#   The 3 chosen distractors always come from 3 distinct scenarios.
#
# -----------------------------------------------------------------------------
# USAGE
# -----------------------------------------------------------------------------
#
#   # Build MCQ questions
#   python quaternary_pipeline.py --build
#   python quaternary_pipeline.py --build --sample 50 --workers 8
#
#   # Run models on questions (evaluation)
#   python quaternary_pipeline.py --run
#   python quaternary_pipeline.py --run --workers 5 --sample 20
#
#   # Summarize results
#   python quaternary_pipeline.py --summary
#
#   # Run all three stages at once
#   python quaternary_pipeline.py --build --run --summary
#
# -----------------------------------------------------------------------------
# OUTPUT DIRECTORY STRUCTURE
# -----------------------------------------------------------------------------
#
#   experiments/mcq/
#   ├── embeddings_cache_context.json                              # Task context embeddings cache
#   ├── embeddings_cache_behavior.json                             # Behavior embeddings cache
#   ├── embeddings_cache_intervention.json                         # Intervention embeddings cache
#   └── quaternary/
#       ├── questions/
#       │   └── {cat_id}_case_{n}_{type}_eff{i}_quaternary.json    # Built MCQ questions
#       ├── results/
#       │   └── {question_id}_{model}_{ts}.json                    # Per-model evaluation results
#       └── summary/
#           └── quaternary_summary_{ts}.json                       # Accuracy summary
#
# -----------------------------------------------------------------------------
# QUESTION ID FORMAT
# -----------------------------------------------------------------------------
#
#   {category_id}_case_{case_num}_{response_type}_eff{eff_idx}_quaternary
#
#   category_id   : e.g. 0001-01
#   case_num      : 3-digit zero-padded scenario index (e.g. 001)
#   response_type : immediate | long_term
#   eff_idx       : index of the effective intervention used as the correct answer
#
# =============================================================================


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

SCENARIO_DIR  = os.path.join(PROJECT_ROOT, "data", "pipeline3", "build_scenario")
DIALOGUE_DIR  = os.path.join(PROJECT_ROOT, "data", "pipeline4", "dialogue")
OUT_ROOT      = os.path.join(PROJECT_ROOT, "experiments", "mcq", "quaternary")
QUESTIONS_DIR = os.path.join(OUT_ROOT, "questions")
RESULTS_DIR   = os.path.join(OUT_ROOT, "results")
SUMMARY_DIR   = os.path.join(OUT_ROOT, "summary")
CTX_CACHE     = os.path.join(PROJECT_ROOT, "experiments", "mcq", "embeddings_cache_context.json")
BEHAV_CACHE   = os.path.join(PROJECT_ROOT, "experiments", "mcq", "embeddings_cache_behavior.json")
INTERV_CACHE  = os.path.join(PROJECT_ROOT, "experiments", "mcq", "embeddings_cache_intervention.json")

JUDGE_MODEL_TYPE = "gemini"
JUDGE_MODEL_NAME = "gemini-3.1-pro-preview"
ADAPT_MODEL_TYPE = "gemini"
ADAPT_MODEL_NAME = "gemini-3.1-pro-preview"

RANDOM_SEED      = 42
LETTERS          = ["A", "B", "C", "D"]
TOP_K_CANDIDATES = 10


# ============================================================================================================================
# SCENARIO LOADING
# ============================================================================================================================
def load_scenarios() -> list[dict]:
    """Actionable scenarios only, renumbered to match pipeline4 case_NNN."""
    entries = []
    for f in sorted(glob.glob(os.path.join(SCENARIO_DIR, "*.json"))):
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        category_id    = data["category_id"]
        paper_id       = data.get("paper_id", "")
        persona        = data.get("student_persona", {}) or {}
        challenge_type = (data.get("challenge_categorization", {})
                              .get("primary_challenge", {})
                              .get("type", "Unknown"))
        actionable = [s for s in data.get("scenarios", []) if s.get("is_actionable")]
        for i, scenario in enumerate(actionable, start=1):
            hs      = scenario.get("hidden_strategies", {}) or {}
            content = scenario.get("content", {}) or {}
            task    = content.get("task", {}) or {}
            ctx     = content.get("context", {}) or {}
            cbt     = content.get("challenging_behavior_trigger", {}) or {}
            bp_list = content.get("behavior_patterns") or []

            ctx_parts = [v for k, v in ctx.items()
                         if k != "evidence_basis" and isinstance(v, str) and v.strip()]
            task_context_text = " ".join(filter(None,
                [task.get("content", "")] + ctx_parts
            )).strip()

            behav_parts = [cbt.get("condition", ""), cbt.get("observable_behavior", "")]
            for p in bp_list:
                if isinstance(p, dict):
                    behav_parts.append(p.get("condition", ""))
                    behav_parts.append(p.get("observable_behavior", ""))
            behavior_text = " / ".join(filter(None, (s.strip() for s in behav_parts if s)))

            eff_imm = hs.get("effective_immediate", []) or []
            eff_lt  = hs.get("effective_long_term", []) or []
            entries.append({
                "category_id":         category_id,
                "paper_id":            paper_id,
                "scenario_idx":        i,
                "scenario_id":         scenario.get("scenario_id", ""),
                "challenge_type":      challenge_type,
                "persona":             persona,
                "content":             content,
                "task_context_text":   task_context_text,
                "behavior_text":       behavior_text,
                "effective_immediate": eff_imm,
                "effective_long_term": eff_lt,
            })
    return entries


def _eff_instruction(eff_list: list, idx: int) -> str:
    if not eff_list or idx < 0 or idx >= len(eff_list):
        return ""
    e = eff_list[idx]
    if not isinstance(e, dict):
        return ""
    return (e.get("instruction", "") or "").strip()


_ELICIT_MARKERS = (
    "Provide a direct, immediate response",
    "Describe what long-term strategies and support structures",
)


def load_dialogue(category_id: str, case_num: int, response_type: str) -> str:
    path = os.path.join(
        DIALOGUE_DIR, f"{category_id}_case_{case_num:03d}_{response_type}.txt"
    )
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as fp:
        txt = fp.read().strip()
    # Drop the free-text-response elicitation tail (pipeline-4 leftover): the
    # "(2-3 sentences)/Output Response ONLY:" instructions conflict with the MCQ
    # task. Keep the "YOUR TURN ..." header + question so the immediate vs
    # long-term horizon signal is preserved.
    for marker in _ELICIT_MARKERS:
        i = txt.find(marker)
        if i != -1:
            return txt[:i].rstrip()
    return txt



# ============================================================================================================================
# EMBEDDINGS
# ============================================================================================================================
def _embed_field(entries: list[dict], text_field: str, cache_path: str) -> dict[str, list[float]]:
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as fp:
            cache = json.load(fp)
    to_embed = [e for e in entries
                if e["scenario_id"] not in cache and e[text_field].strip()]
    if to_embed:
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env.local"))
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print(f"Embedding {len(to_embed)} new scenarios on '{text_field}'...")
        BATCH = 64
        for i in range(0, len(to_embed), BATCH):
            batch = to_embed[i:i+BATCH]
            texts = [e[text_field] for e in batch]
            resp  = client.embeddings.create(model="text-embedding-3-large", input=texts)
            for e, item in zip(batch, resp.data):
                cache[e["scenario_id"]] = item.embedding
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as fp:
            json.dump(cache, fp)
    return {e["scenario_id"]: cache[e["scenario_id"]]
            for e in entries if e["scenario_id"] in cache}


def _embed_intervention(entries: list[dict], cache_path: str) -> dict[str, list[float]]:
    """Embed each effective intervention. Cache key: '{scenario_id}_{rt}_{idx}'."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as fp:
            cache = json.load(fp)
    pairs: list[tuple[str, str]] = []
    for e in entries:
        for rt, eff_key in [("immediate", "effective_immediate"),
                            ("long_term",  "effective_long_term")]:
            for idx in range(len(e[eff_key])):
                text = _eff_instruction(e[eff_key], idx)
                if not text:
                    continue
                key = f"{e['scenario_id']}_{rt}_{idx}"
                if key not in cache:
                    pairs.append((key, text))
    if pairs:
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env.local"))
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print(f"Embedding {len(pairs)} new interventions...")
        BATCH = 64
        for i in range(0, len(pairs), BATCH):
            batch = pairs[i:i+BATCH]
            texts = [p[1] for p in batch]
            resp  = client.embeddings.create(model="text-embedding-3-large", input=texts)
            for (k, _), item in zip(batch, resp.data):
                cache[k] = item.embedding
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as fp:
            json.dump(cache, fp)
    out = {}
    for e in entries:
        for rt, eff_key in [("immediate", "effective_immediate"),
                            ("long_term",  "effective_long_term")]:
            for idx in range(len(e[eff_key])):
                k = f"{e['scenario_id']}_{rt}_{idx}"
                if k in cache:
                    out[k] = cache[k]
    return out


def build_embeddings(entries: list[dict]) -> tuple[dict, dict, dict]:
    """Returns (ctx_embeddings, behavior_embeddings, intervention_embeddings)."""
    ctx    = _embed_field(entries, "task_context_text", CTX_CACHE)
    behav  = _embed_field(entries, "behavior_text",     BEHAV_CACHE)
    interv = _embed_intervention(entries, INTERV_CACHE)
    print(f"  ctx: {len(ctx)}  |  behavior: {len(behav)}  |  intervention: {len(interv)}")
    return ctx, behav, interv


def _cosine(a, b) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na  = sum(x*x for x in a) ** 0.5
    nb  = sum(x*x for x in b) ** 0.5
    return dot / (na * nb + 1e-9)


def find_top_k(target: dict, entries: list[dict],
               ctx_emb: dict, behav_emb: dict, interv_emb: dict,
               eff_key: str, response_type: str,
               top_k: int = TOP_K_CANDIDATES) -> list[tuple[dict, int]]:
    """Top-K (scenario, eff_idx) candidates ranked by combined score:
        score = ctx_sim × (1 - behav_sim) × (1 - max_interv_sim)
    Every effective intervention of every candidate scenario is scored (not just
    [0]); max_interv_sim is the highest cosine between THAT candidate effective and
    ANY of the target's effective interventions (labeled correct + all the student's
    other effectives), so near-duplicates are pushed out at the ranking stage.
    Each scenario is then represented by its single best-scoring effective, so the
    3 chosen distractors still come from 3 distinct scenarios.
    Excludes same-paper and same-category scenarios.
    Returns a list of (entry, eff_idx) pairs.
    """
    t_ctx   = ctx_emb.get(target["scenario_id"])
    t_behav = behav_emb.get(target["scenario_id"])
    t_ivs   = [
        interv_emb[k] for j in range(len(target[eff_key]))
        if (k := f"{target['scenario_id']}_{response_type}_{j}") in interv_emb
    ]
    if not (t_ctx and t_behav and t_ivs):
        return []
    best_per_scenario: dict[str, tuple[float, dict, int]] = {}
    for e in entries:
        if e["paper_id"] and e["paper_id"] == target["paper_id"]:
            continue
        if e["category_id"] == target["category_id"]:
            continue
        if not e[eff_key]:
            continue
        c_ctx   = ctx_emb.get(e["scenario_id"])
        c_behav = behav_emb.get(e["scenario_id"])
        if not (c_ctx and c_behav):
            continue
        ctx_sim    = _cosine(t_ctx,   c_ctx)
        behav_diff = 1.0 - _cosine(t_behav, c_behav)
        for ci in range(len(e[eff_key])):
            c_iv = interv_emb.get(f"{e['scenario_id']}_{response_type}_{ci}")
            if not c_iv:
                continue
            max_interv_sim = max(_cosine(t_iv, c_iv) for t_iv in t_ivs)
            interv_diff    = 1.0 - max_interv_sim
            score          = ctx_sim * behav_diff * interv_diff
            prev = best_per_scenario.get(e["scenario_id"])
            if prev is None or score > prev[0]:
                best_per_scenario[e["scenario_id"]] = (score, e, ci)
    scored = [(s, (e, ci)) for s, e, ci in best_per_scenario.values()]
    scored.sort(key=lambda x: -x[0])
    return [pair for _, pair in scored[:top_k]]



# ============================================================================================================================
# DISTRACTOR SELECTION
# ============================================================================================================================
def select_distractors(target: dict, dialogue: str, response_type: str,
                       correct_instruction: str,
                       sibling_instructions: list[str],
                       candidates: list[tuple[dict, int]],
                       eff_key: str) -> tuple[list[tuple[dict, int]], list[bool]] | None:
    """LLM judge picks the 3 best-fitting (scenario, eff_idx) candidates (dropping
    off-setting or near-duplicate ones), always returns 3, and flags the slots it
    was forced to fill with a poor-fit candidate. Returns (3 (entry, idx) pairs,
    3 forced-pick bools) or None only on a genuine judge failure."""
    if len(candidates) < 3:
        return None

    persona_text = "\n".join(
        f"  {k}: {v}" for k, v in target["persona"].items() if v
    ) or "  (none)"
    target_context = target["task_context_text"]

    cand_blocks = []
    for i, (c, ci) in enumerate(candidates):
        intervention = _eff_instruction(c[eff_key], ci)
        cand_blocks.append(
            f"[{i}]\n"
            f"  context     : {c['task_context_text'][:240]}\n"
            f"  behavior    : {c['behavior_text'][:240]}\n"
            f"  intervention: {intervention[:320]}"
        )

    siblings_text = (
        "\n".join(f"  - {s}" for s in sibling_instructions)
        if sibling_instructions else "  (none — this is the only effective intervention recorded for this student/RT)"
    )

    vars = {
        "DIALOGUE":              dialogue or "",
        "STUDENT_PERSONA":       persona_text,
        "TARGET_CONTEXT":        target_context,
        "RESPONSE_TYPE":         response_type,
        "CORRECT_INTERVENTION":  correct_instruction,
        "SIBLING_INTERVENTIONS": siblings_text,
        "CANDIDATE_DISTRACTORS": "\n\n".join(cand_blocks),
    }
    raw = generate_llm_response("mcq_select_distractors", vars,
                                JUDGE_MODEL_TYPE, JUDGE_MODEL_NAME)
    if not raw:
        return None
    try:
        out  = extract_json(raw)
        idxs = out.get("kept_idxs") or []
        idxs = [int(x) for x in idxs]
        if len(idxs) != 3 or len(set(idxs)) != 3:
            return None
        if any(not (0 <= i < len(candidates)) for i in idxs):
            return None
        forced = set()
        for x in (out.get("forced_idxs") or []):
            try:
                forced.add(int(x))
            except (TypeError, ValueError):
                pass
        selected     = [candidates[i] for i in idxs]
        forced_flags = [i in forced for i in idxs]
        return selected, forced_flags
    except Exception:
        return None



# ============================================================================================================================
# OPTION ADAPTATION
# ============================================================================================================================
def adapt_options(target: dict, dialogue: str, response_type: str,
                  correct: dict, distractors: list[tuple[dict, int]],
                  eff_key: str) -> dict | None:
    """Surface-align correct + 3 distractors in a single LLM call. Each distractor
    is an (entry, eff_idx) pair. Returns dict with adapted strings + edit metadata,
    or None on failure."""
    def _instr(pair):
        d, di = pair
        return _eff_instruction(d[eff_key], di)

    persona_text = "\n".join(
        f"  {k}: {v}" for k, v in target["persona"].items() if v
    ) or "  (none)"
    target_context = target["task_context_text"]

    d1, d2, d3 = distractors
    vars = {
        "DIALOGUE":                  dialogue or "",
        "TARGET_CONTEXT":            target_context,
        "STUDENT_PERSONA":           persona_text,
        "RESPONSE_TYPE":             response_type,
        "CORRECT_INSTRUCTION":       correct.get("instruction", ""),
        "DISTRACTOR_1_INSTRUCTION":  _instr(d1),
        "DISTRACTOR_2_INSTRUCTION":  _instr(d2),
        "DISTRACTOR_3_INSTRUCTION":  _instr(d3),
    }
    raw = generate_llm_response("mcq_adapt_options", vars,
                                ADAPT_MODEL_TYPE, ADAPT_MODEL_NAME)
    if not raw:
        return None
    try:
        out = extract_json(raw)
        needed = ["correct_adapted", "distractor_1_adapted",
                  "distractor_2_adapted", "distractor_3_adapted"]
        if any(k not in out or not str(out[k]).strip() for k in needed):
            return None
        return out
    except Exception:
        return None



# ============================================================================================================================
# QUESTION BUILDER
# ============================================================================================================================
def build_one(args: tuple) -> dict | None:
    entry, response_type, eff_key, eff_idx, ctx_emb, behav_emb, interv_emb, all_entries = args
    case_num   = entry["scenario_idx"]
    effectives = entry[eff_key]
    if eff_idx < 0 or eff_idx >= len(effectives):
        return None
    correct             = effectives[eff_idx]
    correct_instruction = _eff_instruction(effectives, eff_idx)
    if not correct_instruction:
        return None
    dlg = load_dialogue(entry["category_id"], case_num, response_type)
    if not dlg:
        return None

    sibling_instructions = [
        _eff_instruction(effectives, j)
        for j in range(len(effectives)) if j != eff_idx
    ]
    sibling_instructions = [s for s in sibling_instructions if s]

    candidates = find_top_k(entry, all_entries,
                            ctx_emb, behav_emb, interv_emb,
                            eff_key, response_type)
    if len(candidates) < 3:
        return None

    sel = select_distractors(
        entry, dlg, response_type,
        correct_instruction, sibling_instructions,
        candidates, eff_key,
    )
    if sel is None:
        return None
    distractors, forced_flags = sel

    dist_entries = [d for d, _ in distractors]
    raw_distractor_instructions = [
        _eff_instruction(d[eff_key], di) for d, di in distractors
    ]
    if any(not t for t in raw_distractor_instructions):
        return None

    adapted = adapt_options(entry, dlg, response_type, correct, distractors, eff_key)
    if adapted is None:
        return None

    correct_text = adapted["correct_adapted"]
    distractor_texts = [
        adapted["distractor_1_adapted"],
        adapted["distractor_2_adapted"],
        adapted["distractor_3_adapted"],
    ]

    rng = random.Random(
        f"{entry['category_id']}_{case_num}_{response_type}_eff{eff_idx}_{RANDOM_SEED}"
    )
    dist_src_idxs = [di for _, di in distractors]
    pool = [
        ("correct",      correct_text,        entry,           correct_instruction,           False,          eff_idx),
        ("distractor_0", distractor_texts[0], dist_entries[0], raw_distractor_instructions[0], forced_flags[0], dist_src_idxs[0]),
        ("distractor_1", distractor_texts[1], dist_entries[1], raw_distractor_instructions[1], forced_flags[1], dist_src_idxs[1]),
        ("distractor_2", distractor_texts[2], dist_entries[2], raw_distractor_instructions[2], forced_flags[2], dist_src_idxs[2]),
    ]
    rng.shuffle(pool)

    options, sources = {}, {}
    correct_letter = None
    n_forced_picks = 0
    for letter, (role, text, src, raw_instr, forced, src_eff_idx) in zip(LETTERS, pool):
        options[letter] = text
        sources[letter] = {
            "role":            role,
            "scenario_id":     src["scenario_id"],
            "category_id":     src["category_id"],
            "source_eff_idx":  src_eff_idx,
            "raw_instruction": raw_instr,
            "forced_pick":     bool(forced),
        }
        if forced:
            n_forced_picks += 1
        if role == "correct":
            correct_letter = letter

    return {
        "question_id":     f"{entry['category_id']}_case_{case_num:03d}_{response_type}_eff{eff_idx}_quaternary",
        "category_id":     entry["category_id"],
        "scenario_id":     entry["scenario_id"],
        "case_num":        case_num,
        "response_type":   response_type,
        "effective_idx":   eff_idx,
        "n_siblings":      len(sibling_instructions),
        "n_forced_picks":  n_forced_picks,
        "has_forced_pick": n_forced_picks > 0,
        "difficulty":      "quaternary",
        "challenge_type":  entry["challenge_type"],
        "dialogue":        dlg,
        "student_context": entry["persona"].get("description", ""),
        "options":         options,
        "option_sources":  sources,
        "correct_answer":  correct_letter,
        "edit_level":      adapted.get("edit_level", "unknown"),
        "edit_notes":      adapted.get("edit_notes", ""),
    }


def build(sample: int = 0, workers: int = 8):
    os.makedirs(QUESTIONS_DIR, exist_ok=True)
    entries = load_scenarios()
    print(f"Loaded {len(entries)} actionable scenarios.")
    ctx_emb, behav_emb, interv_emb = build_embeddings(entries)

    jobs = []
    for e in entries:
        for rt, eff_key in [("immediate", "effective_immediate"),
                            ("long_term", "effective_long_term")]:
            if not e[eff_key]:
                continue
            if not load_dialogue(e["category_id"], e["scenario_idx"], rt):
                continue
            for eff_idx in range(len(e[eff_key])):
                if not _eff_instruction(e[eff_key], eff_idx):
                    continue
                jobs.append(
                    (e, rt, eff_key, eff_idx, ctx_emb, behav_emb, interv_emb, entries)
                )
    print(f"Producible: {len(jobs)} (one job per effective intervention)")

    if sample > 0:
        random.Random(0).shuffle(jobs)
        jobs = jobs[:sample]
        print(f"SAMPLE mode: {len(jobs)} questions.")

    successes = failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i, q in enumerate(pool.map(build_one, jobs), 1):
            if q is None:
                failures += 1
                continue
            successes += 1
            with open(os.path.join(QUESTIONS_DIR, f"{q['question_id']}.json"),
                      "w", encoding="utf-8") as fp:
                json.dump(q, fp, indent=2, ensure_ascii=False)
            if i % 25 == 0 or sample > 0:
                print(f"  [{i}/{len(jobs)}] {q['question_id']}")
    print(f"\nDone. successes={successes}, failures={failures}")



# ============================================================================================================================
# RUN
# ============================================================================================================================
def _result_exists(question_id: str, model_name: str) -> bool:
    safe = model_name.replace("/", "-")
    return bool(glob.glob(os.path.join(RESULTS_DIR, f"{question_id}_{safe}_*.json")))


def _parse_answer(raw: str) -> tuple[str, str]:
    try:
        parsed = extract_json(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        ans = str(parsed.get("answer", "")).strip().upper()
        rsn = str(parsed.get("reasoning", "")).strip()
        if ans in LETTERS:
            return ans, rsn
    m = re.search(r'\b([ABCD])\b', raw.upper())
    return (m.group(1) if m else ""), raw.strip()


def run_one(args: tuple) -> dict | None:
    q, model_type, model_name, ts, idx, total = args
    qid  = q["question_id"]
    safe = model_name.replace("/", "-")
    if _result_exists(qid, model_name):
        return None
    options_text = "\n".join(f"{letter}) {text}" for letter, text in q["options"].items())
    print(f"  [{idx}/{total}] {qid} @ {model_name}", end="", flush=True)
    try:
        raw = generate_llm_response(
            "mcq_evaluate",
            {"DIALOGUE": q["dialogue"], "OPTIONS": options_text},
            model_type, model_name, use_api_defaults=True,
        )
    except Exception as e:
        print(f" [ERROR: {type(e).__name__}: {str(e)[:80]}]")
        return None
    if raw is None:
        print(" [FAILED]")
        return None
    answer, reasoning = _parse_answer(raw)
    is_correct = answer == q["correct_answer"]
    print(f" → {answer} ({'✓' if is_correct else '✗'})")

    result = {
        "question_id":    qid,
        "category_id":    q["category_id"],
        "case_num":       q["case_num"],
        "response_type":  q["response_type"],
        "difficulty":     q["difficulty"],
        "challenge_type": q["challenge_type"],
        "model":          model_name,
        "correct_answer": q["correct_answer"],
        "model_answer":   answer,
        "is_correct":     is_correct,
        "reasoning":      reasoning,
        "evaluated_at":   datetime.now().isoformat(),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{qid}_{safe}_{ts}.json"),
              "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
    return result


def run(workers: int = 5, config_path: str = "config/config_response.json",
        sample: int = 0):
    questions = []
    for f in sorted(glob.glob(os.path.join(QUESTIONS_DIR, "*.json"))):
        with open(f, encoding="utf-8") as fp:
            questions.append(json.load(fp))
    if not questions:
        raise FileNotFoundError(f"No questions in {QUESTIONS_DIR}. Run --build first.")
    if sample > 0:
        random.Random(0).shuffle(questions)
        questions = questions[:sample]
        questions.sort(key=lambda q: q["question_id"])
        print(f"SAMPLE mode: {len(questions)} questions (deterministic random subset, seed 0).")
    else:
        print(f"Loaded {len(questions)} questions.")

    with open(config_path) as fp:
        models = json.load(fp)["models_to_test"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for m in models:
        name, mtype = m["name"], m["model_type"]
        print(f"\nRunning [{name}] ({workers} workers)...")
        jobs = [(q, mtype, name, ts, i, len(questions)) for i, q in enumerate(questions, 1)]
        results, skipped = [], 0
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                for r in pool.map(run_one, jobs):
                    if r is None:
                        skipped += 1
                    else:
                        results.append(r)
        except Exception as e:
            print(f"\n[{name}] aborted: {type(e).__name__}: {str(e)[:120]}")
            continue
        correct = sum(r["is_correct"] for r in results)
        total   = len(results)
        if total:
            print(f"[{name}] Done: {correct}/{total} correct ({correct/total:.1%}). Skipped {skipped}.")
        else:
            print(f"[{name}] No new results. Skipped {skipped}.")



# ============================================================================================================================
# SUMMARY
# ============================================================================================================================
def _acc(rs):
    n = len(rs)
    return {"n": n, "accuracy": round(sum(r["is_correct"] for r in rs)/n, 4) if n else None}


def summarize():
    results = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        try:
            with open(f, encoding="utf-8") as fp:
                results.append(json.load(fp))
        except Exception:
            pass
    if not results:
        print("No results found.")
        return

    by_model: defaultdict = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    summary = {"n_total": len(results), "by_model": {}}
    for model, rs in by_model.items():
        by_ct, by_rt = defaultdict(list), defaultdict(list)
        for r in rs:
            by_ct[r["challenge_type"]].append(r)
            by_rt[r["response_type"]].append(r)
        summary["by_model"][model] = {
            **_acc(rs),
            "by_challenge_type": {ct: _acc(v) for ct, v in by_ct.items()},
            "by_response_type":  {rt: _acc(v) for rt, v in by_rt.items()},
        }

    os.makedirs(SUMMARY_DIR, exist_ok=True)
    out = os.path.join(SUMMARY_DIR,
                       f"quaternary_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    print(f"Summary saved: {out}")
    for model, s in summary["by_model"].items():
        if s['accuracy'] is not None:
            print(f"  {model}: {s['n']} questions, {s['accuracy']*100:.1f}% accuracy")
        else:
            print(f"  {model}: n={s['n']}")



# ============================================================================================================================
# MAIN
# ============================================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build",   action="store_true")
    ap.add_argument("--run",     action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--sample",  type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config",  default="config/config_response.json")
    args = ap.parse_args()

    if not any([args.build, args.run, args.summary]):
        ap.error("Pick at least one of --build, --run, --summary.")

    if args.build:   build(sample=args.sample, workers=args.workers)
    if args.run:     run(workers=args.workers, config_path=args.config, sample=args.sample)
    if args.summary: summarize()


if __name__ == "__main__":
    main()
