import argparse
import json
import random
import os
import re
from glob import glob

from utils.analyze_paper_utils import extract_text_from_pdf
from utils.prompt_utils import extract_json_from_response
from utils.llm_utils import generate_llm_response, get_pipeline_model
from utils.output_utils import save_json_output
from utils.generate_dialogue_utils import generate_all_dialogue_files
from utils.generate_response_utils import collect_model_responses
from utils.pipeline_utils import auto_resolve_dependencies, resolve_pipeline_output


# =============================================================================
# main.py — Main Dialogue Construction Pipeline
# =============================================================================
#
# Accepts pedagogical case-study PDFs and runs Pipelines 1–5 sequentially
# to automatically construct dialogue datasets for evaluating LLM tutors.
#
# Each stage is automatically skipped if its output already exists.
#
# -----------------------------------------------------------------------------
# PIPELINE FLOW
# -----------------------------------------------------------------------------
#
#   [Pipeline 1-A]  screen_paper               : Pre-filter paper (rejected if any of 3 dims < 3)
#   [Pipeline 1-B]  analyze_paper              : Extract student categories & assess eligibility
#   [Pipeline 2-A]  extract_excerpt_learner    : Extract learner evidence (behavior/cognitive/affective)
#   [Pipeline 2-B]  extract_excerpt_instruction: Extract instructional evidence (immediate/long-term)
#   [Pipeline 3]    build_scenario             : Build per-category scenarios + hidden_strategies
#   [Pipeline 4]    synthesize_dialogue        : Synthesize multi-turn dialogues → save as .txt
#   [Pipeline 5]    collect_model_responses    : Collect responses from all models in config_response.json
#
# -----------------------------------------------------------------------------
# USAGE
# -----------------------------------------------------------------------------
#
#   # Basic run
#   python main.py data/docs/0001.pdf
#
#   # Specify LLM engine
#   python main.py data/docs/0001.pdf --model openai
#
#   # Stop after a specific pipeline stage
#   python main.py data/docs/0001.pdf --pipeline1-only
#   python main.py data/docs/0001.pdf --pipeline2-only
#   python main.py data/docs/0001.pdf --pipeline3-only
#   python main.py data/docs/0001.pdf --pipeline4-only
#   python main.py data/docs/0001.pdf --pipeline5-only
#
#   # Resume from a later stage (reusing existing outputs)
#   python main.py --pipeline1-output 0001       # reuse P1 output, run from P2
#   python main.py --pipeline2-output 0001       # reuse P2 output, run from P3
#   python main.py --pipeline3-output 0001-01    # reuse P3 output, run from P4
#   python main.py --pipeline4-output 0001-01    # reuse P4 output, run P5 only
#
#   # Prompt user confirmation before each stage
#   python main.py data/docs/0001.pdf --check true
#
#   # Verbose output
#   python main.py data/docs/0001.pdf --verbose
#
# -----------------------------------------------------------------------------
# OUTPUT DIRECTORY STRUCTURE
# -----------------------------------------------------------------------------
#
#   data/
#   ├── logs/
#   │   └── {paper_id}.txt                                        # Pipeline run log
#   │
#   ├── pipeline1/
#   │   ├── screen_paper/
#   │   │   └── screen_paper_{paper_id}_{ts}.json                 # Pre-filter result
#   │   └── analyze_paper/
#   │       └── analyze_paper_{paper_id}_{ts}.json                # Category analysis result
#   │
#   ├── pipeline2/
#   │   ├── extract_excerpt_learner/
#   │   │   └── extract_excerpt_learner_{paper_id}_{ts}.json      # Learner excerpts
#   │   └── extract_excerpt_instruction/
#   │       └── extract_excerpt_instruction_{paper_id}_{ts}.json  # Instructional excerpts
#   │
#   ├── pipeline3/
#   │   └── build_scenario/
#   │       └── build_scenario_{cat_id}_{ts}.json                 # Scenarios + hidden_strategies
#   │
#   ├── pipeline4/
#   │   ├── synthesize_dialogue/
#   │   │   └── synthesize_dialogue_{cat_id}_{ts}.json            # Dialogue synthesis result (JSON)
#   │   └── dialogue/
#   │       └── {cat_id}_case_{n}_{type}.txt                      # Evaluation dialogue files (immediate / long_term)
#   │
#   └── pipeline5/
#       └── response/
#           └── {cat_id}_case_{n}_{type}_{model}.txt              # Per-model response files
#
# -----------------------------------------------------------------------------
# FILTER THRESHOLDS  (config/config_filter.json)
# -----------------------------------------------------------------------------
#
#   screen_paper  : Paper rejected if any of the 3 dims
#                   (student_behavior / learning_context / teacher_actions) scores below 3
#   analyze_paper : Category excluded if any of the 3 dims scores 3 or below
#   extract_excerpt: Category excluded if learner or instruction excerpt count is 0
#
# =============================================================================


# ============================================================================================================================
# LOGGER
# ============================================================================================================================
class PipelineLogger:
    def __init__(self, paper_id: str):
        self.paper_id = paper_id
        self.log_dir = os.path.join("data", "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, f"{paper_id}.txt")
        # Write header once
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"[LOG] paper_id={paper_id}\n")
            f.write(f"{'='*60}\n")

    def write(self, text: str):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def section(self, title: str):
        self.write(f"\n--- {title} ---")

    # Pipeline 1: category scores
    def log_pipeline1(self, category_list: list):
        self.section("Pipeline 1: Analyze Paper")
        for cat in category_list:
            cat_id = cat.get("category_id", "N/A")
            name = cat.get("name", "N/A")
            elig = cat.get("extraction_eligibility", {})
            self.write(f"  [{cat_id}] {name}")
            for dim in ["student_behavior", "learning_context", "teacher_actions"]:
                self.write(f"    {dim:<20}: {elig.get(dim, {}).get('score', 'N/A')}/5")

    # Pipeline 2: excerpt counts per category
    def log_pipeline2(self, learner_data: dict, instruction_data: dict, filtered_ids: list = None):
        self.section("Pipeline 2: Extract Excerpt")
        learner_summary = learner_data.get("extraction_summary", {})
        instruction_summary = instruction_data.get("extraction_summary", {})
        learner_epc = learner_summary.get("excerpts_per_category", {})
        instruction_epc = instruction_summary.get("excerpts_per_category", {})
        all_cat_ids = set(list(learner_epc.keys()) + list(instruction_epc.keys()))
        for cat_id in sorted(all_cat_ids):
            l_count = learner_epc.get(cat_id, 0)
            i_count = instruction_epc.get(cat_id, 0)
            is_filtered = filtered_ids and cat_id in filtered_ids
            tag = " [FILTERED]" if is_filtered else ""
            self.write(f"  [{cat_id}]{tag}")
            self.write(f"    learner excerpts     : {l_count}")
            self.write(f"    instruction excerpts : {i_count}")

    # Pipeline 3: scenario title, task content, student persona
    def log_pipeline3(self, scenario_data: dict):
        self.section(f"Pipeline 3: Build Scenario [{scenario_data.get('category_id', 'N/A')}]")
        persona = scenario_data.get("student_persona", {})
        self.write(f"  Student Persona Description:")
        self.write(f"    {persona.get('description', 'N/A')[:200]}")
        for scenario in scenario_data.get("scenarios", []):
            s_id = scenario.get("scenario_id", "N/A")
            title = scenario.get("title", "N/A")
            task_content = scenario.get("content", {}).get("task", {}).get("content", "N/A")
            self.write(f"  [{s_id}] {title}")
            self.write(f"    task               : {task_content[:120]}")

    # Pipeline 4: dialogue summary per case
    def log_pipeline4(self, dialogue_data: dict):
        self.section(f"Pipeline 4: Synthesize Dialogue [{dialogue_data.get('category_id', 'N/A')}]")
        for dlg in dialogue_data.get("dialogues", []):
            case_id = dlg.get("case_id", "N/A")
            turns = dlg.get("dialogue", [])
            challenging = next((t for t in reversed(turns) if t.get("is_challenging_behavior")), None)
            self.write(f"  [{case_id}]")
            self.write(f"    turns          : {len(turns)}")
            self.write(f"    minimal_context: {str(dlg.get('minimal_context', 'N/A'))[:120]}")
            if challenging:
                self.write(f"    challenging    : {challenging.get('message', '')[:120]}")



# ============================================================================================================================
# HELPERS
# ============================================================================================================================
def load_scenario_config(config_path: str = "config/config_filter.json") -> dict:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f).get("scenario", {})
    except Exception:
        return {}


def log_pipeline1_scores(pipeline1_data: dict, logger: PipelineLogger):
    if logger:
        logger.log_pipeline1(pipeline1_data.get("category_list", []))


def _epc_lookup(epc: dict, cat_id: str) -> int:
    """Look up excerpt count by cat_id, falling back to the bare NNNN-NN suffix."""
    if cat_id in epc:
        return epc[cat_id]
    m = re.search(r'\d{4}-\d{2}$', cat_id)
    if m:
        return epc.get(m.group(0), 0)
    return 0


def annotate_pipeline2(learner_data: dict, instruction_data: dict, category_list: list,
                       logger: PipelineLogger = None) -> list:
    """Annotates each category with quality_gate info. Returns all categories (nothing dropped)."""
    learner_epc = learner_data.get("extraction_summary", {}).get("excerpts_per_category", {})
    instr_epc = instruction_data.get("extraction_summary", {}).get("excerpts_per_category", {})

    below_threshold_ids = []
    for cat in category_list:
        cat_id = cat.get("category_id")
        l_count = _epc_lookup(learner_epc, cat_id)
        i_count = _epc_lookup(instr_epc, cat_id)
        passed = l_count > 0 and i_count > 0
        if not passed:
            print(f"  [P2 FILTERED] Category {cat_id}: learner={l_count}, instruction={i_count} (need > 0)")
        cat["quality_gate"] = {
            "pipeline2_passed": passed,
            "learner_excerpt_count": l_count,
            "instruction_excerpt_count": i_count,
        }
        if not passed:
            below_threshold_ids.append(cat_id)

    if logger:
        logger.log_pipeline2(learner_data, instruction_data, below_threshold_ids)
    return category_list



# ============================================================================================================================
# PIPELINE
# ============================================================================================================================
def load_turn_config(config_path: str = "config/config_dialogue_turn.json") -> tuple[int, int]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("min_turn", 3), cfg.get("max_turn", 3)
    except Exception:
        return 3, 3
min_turn, max_turn = load_turn_config()


def _derive_study_group(pdf_path: str) -> str | None:
    """Return study group name if pdf_path is under docs/{study_group}/."""
    abs_path = os.path.abspath(pdf_path)
    parent_name     = os.path.basename(os.path.dirname(abs_path))
    grandparent_name = os.path.basename(os.path.dirname(os.path.dirname(abs_path)))
    if grandparent_name.lower() == "docs" and not re.match(r"^\d+$", parent_name):
        return parent_name
    return None


class StudentSimulatorPipeline:
    def __init__(self, args):
        self.args = args
        self._p1 = get_pipeline_model("pipeline1")
        self._p2 = get_pipeline_model("pipeline2")
        self._p3 = get_pipeline_model("pipeline3")
        self._p4 = get_pipeline_model("pipeline4")

    # Pipeline 0: Screen Paper (cheap pre-filter before full extraction)
    def screen_paper(self, texts: str):
        return generate_llm_response("screen_paper", {"TEXT": texts}, **self._p1)

    # Pipeline 1: Analyze Paper
    def analyze_paper(self, texts: str, paper_id: str = None):
        return generate_llm_response("analyze_paper", {"TEXT": texts, "ID": paper_id or ""}, **self._p1)

    # Pipeline 2: Extract Excerpt
    def extract_excerpt(self, path: str, data: list, excluded_pages: list, paper_id: str = None):
        texts = extract_text_from_pdf(path, excluded_pages)
        pid  = paper_id or path.replace("\\", "/").split("/")[-1].replace(".pdf", "")
        data = str([{k: category[k] for k in ["category_id", "name", "label", "demographics"] if k in category} for category in data])
        if not texts:
            return None
        learner_response = generate_llm_response(
            "extract_excerpt_learner",
            {"TEXT": texts, "ID": pid, "CATEGORIES": data},
            **self._p2
        )
        instruction_response = generate_llm_response(
            "extract_excerpt_instruction",
            {
                "TEXT": texts,
                "ID": pid,
                "CATEGORIES": data,
                "LEARNER_EXCERPTS": str(extract_json_from_response(learner_response).get("learner_excerpts", []))
            },
            **self._p2
        )
        return learner_response, instruction_response

    # Pipeline 3: Build Scenario
    def build_scenario(self, paper_id: str, category: dict, learner_excerpts: dict, instruction_excerpts: dict, num_scenarios: int = 3):
        if not learner_excerpts or not instruction_excerpts:
            return
        category_id = category["category_id"]
        bare_suffix = re.search(r'\d{4}-\d{2}$', category_id)
        _cat_ids = {category_id}
        if bare_suffix:
            _cat_ids.add(bare_suffix.group(0))

        filtered_learner_excerpts = [
            excerpt for excerpt in learner_excerpts.get("learner_excerpts", [])
            if _cat_ids & set(excerpt.get("applicable_categories", []))
        ]
        filtered_student_quotes = [
            quote for quote in learner_excerpts.get("student_quotes", [])
            if _cat_ids & set(quote.get("applicable_categories", []))
        ]
        learner_excerpt_data = {
            "learner_excerpts": filtered_learner_excerpts,
            "student_quotes": filtered_student_quotes
        }
        filtered_instructional_excerpts = [
            excerpt for excerpt in instruction_excerpts.get("instructional_excerpts", [])
            if _cat_ids & set(excerpt.get("applicable_categories", []))
        ]
        instruction_excerpt_data = {
            "instructional_excerpts": filtered_instructional_excerpts
        }
        return generate_llm_response(
            "build_scenario",
            {
                "ID": paper_id,
                "CATEGORY": str(category),
                "CATEGORY_ID": category_id,
                "LEARNER_EXCERPT": str(learner_excerpt_data),
                "INSTRUCTION_EXCERPT": str(instruction_excerpt_data),
                "NUM_SCENARIOS": str(num_scenarios)
            },
            **self._p3
        )

    # Pipeline 4: Synthesize Dialogue
    def synthesize_dialogue(self, paper_id: str, category: dict,
                        scenario_data: dict, individual_scenario: dict,
                        evidence: dict, turn: int = 3):
        if not individual_scenario:
            return
        scenario_visible = {
            "scenario_id": individual_scenario.get("scenario_id"),
            "title": individual_scenario.get("title"),
            "content": individual_scenario.get("content"),
        }
        category_context = {
            "category_id": category.get("category_id"),
            "name": category.get("name"),
            "label": category.get("label"),
            "categorization_reasoning": category.get("categorization_reasoning"),
            "demographics": category.get("demographics"),
        }
        return generate_llm_response(
            "synthesize_dialogue",
            {
                "ID": paper_id,
                "CATEGORY": str(category_context),
                "CATEGORY_ID": category["category_id"],
                "SCENARIO": str(scenario_visible),
                "STUDENT_PERSONA": str(scenario_data.get("student_persona", {})),
                "EVIDENCE": str(evidence),
                "TURN": str(turn),
            },
            **self._p4
        )



# ============================================================================================================================
# VERBOSE OUTPUT
# ============================================================================================================================
def print_pipeline1_verbose(data):
    meta = data.get("metadata", {})
    scope = data.get("research_scope", {})
    categories = data.get("category_list", [])

    print(f"\nTitle  : {meta.get('title', 'N/A')}")
    print(f"Author : {meta.get('author', 'N/A')} ({meta.get('year', 'N/A')})")
    print(f"Scope  : {scope.get('education_level', 'N/A')} / {scope.get('discipline', 'N/A')} / {scope.get('country', 'N/A') or 'country not specified'}")
    print(f"Method : {meta.get('research_method', 'N/A')[:120]}")

    print(f"\nCategories found: {len(categories)}")
    for idx, cat in enumerate(categories, 1):
        eligibility = cat.get("extraction_eligibility", {})
        print(f"  [{idx}] {cat.get('category_id', 'N/A')} | {cat.get('name', 'N/A')}")
        print(f"       Labels : {', '.join(cat.get('label', [])) or 'none'}")
        for dim in ["student_behavior", "learning_context", "teacher_actions"]:
            score = eligibility.get(dim, {}).get("score", "N/A")
            print(f"         {dim}: {score}/5")


def print_pipeline2_verbose(learner_data, instruction_data, pipeline1_data):
    # Learner
    summary1 = learner_data.get("extraction_summary", {})
    print(f"\n[Learner Excerpts]")
    print(f"  Total excerpts : {summary1.get('total_excerpts', 0)}")
    print(f"  Total quotes   : {summary1.get('total_quotes', 0)}")

    epc = summary1.get("excerpts_per_category", {})
    categories = {c["category_id"]: c["name"] for c in pipeline1_data.get("category_list", [])}
    for cat_id, count in epc.items():
        print(f"    {cat_id} ({categories.get(cat_id, 'unknown')}): {count} excerpts")

    evc = summary1.get("excerpt_verbatim_check", {})
    if evc:
        total = len(evc)
        verbatim = sum(1 for v in evc.values() if v)
        rate = verbatim / total if total > 0 else 0
        print(f"  Verbatim rate  : {verbatim}/{total} ({rate:.1%})")
        if rate < 0.9:
            print("  [WARNING] Verbatim rate below 90%")

    # Instruction
    summary2 = instruction_data.get("extraction_summary", {})
    print(f"\n[Instructional Excerpts]")
    print(f"  Total excerpts : {summary2.get('total_excerpts', 0)}")

    epc2 = summary2.get("excerpts_per_category", {})
    for cat_id, count in epc2.items():
        print(f"    {cat_id} ({categories.get(cat_id, 'unknown')}): {count} excerpts")

    evc2 = summary2.get("excerpt_verbatim_check", {})
    if evc2:
        total = len(evc2)
        verbatim = sum(1 for v in evc2.values() if v)
        rate = verbatim / total if total > 0 else 0
        print(f"  Verbatim rate  : {verbatim}/{total} ({rate:.1%})")
        if rate < 0.9:
            print("  [WARNING] Verbatim rate below 90%")

    linking = summary2.get("linking_summary", {})
    if linking:
        print(f"\n[Linking Summary]")
        print(f"  Total / Strong / Moderate / Weak : {linking.get('total_links', 0)} / {linking.get('strong_links', 0)} / {linking.get('moderate_links', 0)} / {linking.get('weak_links', 0)}")
        orphaned_l = linking.get("orphaned_learner_excerpts", [])
        orphaned_i = linking.get("orphaned_instruction_excerpts", [])
        if orphaned_l:
            print(f"  [WARNING] Orphaned learner excerpts ({len(orphaned_l)}): {', '.join(orphaned_l[:3])}" + (" ..." if len(orphaned_l) > 3 else ""))
        if orphaned_i:
            print(f"  [WARNING] Orphaned instruction excerpts ({len(orphaned_i)}): {', '.join(orphaned_i[:3])}" + (" ..." if len(orphaned_i) > 3 else ""))


def print_pipeline3_verbose(data):
    print(f"\nCategory : {data.get('category_id', 'N/A')} | {data.get('name', 'N/A')}")

    persona = data.get("student_persona", {})
    if persona:
        print(f"\n[Student Persona]")
        print(f"  Behavioral  : {persona.get('behavioral_patterns', 'N/A')[:120]}")
        print(f"  Triggers    : {persona.get('unique_triggers_and_needs', 'N/A')[:120]}")
        print(f"  Approach    : {persona.get('approach_effectiveness', 'N/A')[:120]}")

    scenarios = data.get("scenarios", [])
    print(f"\n[Scenarios] {len(scenarios)} total")
    for i, s in enumerate(scenarios, 1):
        print(f"\n  [{i}] {s.get('scenario_id', 'N/A')} - {s.get('title', 'Untitled')}")
        content = s.get("content", {})
        task = content.get("task", {})
        ctx = content.get("context", {})
        trigger = content.get("challenging_behavior_trigger", {})
        patterns = content.get("behavior_patterns", [])
        hidden = s.get("hidden_strategies", {})

        print(f"    Task       : {task.get('content', 'N/A')[:100]}")
        print(f"    Materials  : {ctx.get('materials', 'N/A')[:80]}")
        print(f"    Setting    : {ctx.get('social_setting', 'N/A')[:80]}")
        print(f"    Trigger    : {trigger.get('condition', 'N/A')[:100]}")
        print(f"    Behaviors  : {len(patterns)} pattern(s)")
        print(f"    Strategies : immediate={len(hidden.get('effective_immediate', []))} / long_term={len(hidden.get('effective_long_term', []))} / ineffective={len(hidden.get('ineffective', []))}")

    challenges = data.get("challenge_categorization", {})
    if challenges:
        primary = challenges.get("primary_challenge", {})
        secondary = challenges.get("secondary_challenge", {})
        print(f"\n[Challenge Categorization]")
        print(f"  Primary   : {primary.get('type', 'N/A')} - {primary.get('evidence_summary', 'N/A')[:80]}")
        if secondary.get("type"):
            print(f"  Secondary : {secondary.get('type', 'N/A')} - {secondary.get('evidence_summary', 'N/A')[:80]}")


def print_pipeline4_verbose(data):
    print(f"\nCategory : {data.get('category_id', 'N/A')} | {data.get('name', 'N/A')}")

    dialogues = data.get("dialogues", [])
    print(f"\n[Dialogues] {len(dialogues)} total")
    for idx, dlg in enumerate(dialogues, 1):
        print(f"\n  [{idx}] {dlg.get('case_id', 'N/A')}")
        turns = dlg.get("dialogue", [])
        print(f"    Turns : {len(turns)}")
        for turn in turns[:2]:
            speaker = turn.get("speaker", "N/A")
            msg = turn.get("message", "")[:80]
            print(f"      Turn {turn.get('turn_id', '?')} ({speaker}): {msg}")
        if len(turns) > 2:
            final = turns[-1]
            if final.get("is_challenging_behavior"):
                print(f"      ...")
                print(f"      Turn {final.get('turn_id', '?')} (student - CHALLENGING): {final.get('message', '')[:80]}")

        print(f"    minimal_context: {str(dlg.get('minimal_context', 'N/A'))[:120]}")



# ============================================================================================================================
# MAIN
# ============================================================================================================================
def main():
    parser = argparse.ArgumentParser(description="Construct datasets for student simulator from pedagogy literature PDFs.")
    parser.add_argument("path", nargs="?", help="PDF file path or directory")
    parser.add_argument("--model", choices=["auto", "openai", "gemini"], default="auto", help="LLM model to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--pipeline1-output", type=str, help="Path to pipeline 1 output JSON (skips pipeline 1)")
    parser.add_argument("--pipeline2-output", type=str, help="Path to pipeline 2 output JSON (skips pipeline 2)")
    parser.add_argument("--pipeline2-output-learner", type=str, help="Path to pipeline 2 learner output JSON")
    parser.add_argument("--pipeline2-output-instruction", type=str, help="Path to pipeline 2 instruction output JSON")
    parser.add_argument("--pipeline3-output", type=str, help="Path to pipeline 3 output JSON (skips pipeline 3)")
    parser.add_argument("--pipeline4-output", type=str, help="Path to pipeline 4 output JSON (skips pipeline 4)")
    parser.add_argument("--check", type=str, default="false", choices=["true", "false"],
                        help="If true, prompt user to confirm before each pipeline (default: false)")
    parser.add_argument("--response-config", type=str, default="config/config_response.json",
                        help="Config file for response collection (default: config/config_response.json)")
    parser.add_argument("--filter-config", type=str, default="config/config_filter.json",
                        help="Path to filter config JSON (default: config/config_filter.json)")
    parser.add_argument("--pipeline1-only", action="store_true",
                        help="Stop after Pipeline 1 (screen + analyze).")
    parser.add_argument("--pipeline2-only", action="store_true",
                        help="Stop after Pipeline 2 (extract excerpt).")
    parser.add_argument("--pipeline3-only", action="store_true",
                        help="Stop after Pipeline 3 (build scenario).")
    parser.add_argument("--pipeline4-only", action="store_true",
                        help="Stop after Pipeline 4 (synthesize dialogue).")
    parser.add_argument("--pipeline5-only", action="store_true",
                        help="Stop after Pipeline 5 (generate responses).")

    args = parser.parse_args()
    use_check = args.check.lower() == "true"
    scenario_cfg = load_scenario_config(args.filter_config)
    num_scenarios = scenario_cfg.get("num_scenarios", 3)

    if not auto_resolve_dependencies(args):
        print("Error: Could not resolve pipeline dependencies")
        return

    pipeline = StudentSimulatorPipeline(args)
    path = args.path
    pipeline1_paper_data = None
    pipeline2_excerpt_data = []
    pipeline3_scenario_data = []
    pipeline4_dialogue_data = []
    effective_paper_id = None   # set once path is known; includes study-group prefix if applicable


    def get_paper_id(p1_data=None, fallback_path=None):
        if p1_data:
            return p1_data.get("id", "unknown")
        if fallback_path:
            return fallback_path.replace("\\", "/").split("/")[-1].replace(".pdf", "")
        return "unknown"

    logger = None


    # ── Pipeline 1: Analyze Paper ─────────────────────────────────────────────
    if hasattr(args, 'pipeline1_output') and args.pipeline1_output:
        print(f">>> Loading pipeline 1 output: {args.pipeline1_output}")
        try:
            with open(args.pipeline1_output, "r", encoding="utf-8") as f:
                pipeline1_paper_data = json.load(f)
            print(f">>> Pipeline 1 loaded")
        except Exception as e:
            print(f"Error loading pipeline 1 output: {e}")
            return
    else:
        if not path:
            path = input("PDF file path: ").strip()

        _filename_id   = path.replace("\\", "/").split("/")[-1].replace(".pdf", "")
        _study_group   = _derive_study_group(path)
        effective_paper_id = f"{_study_group}_{_filename_id}" if _study_group else _filename_id

        # Extract PDF text once — shared by screen and analyze_paper
        pdf_texts = extract_text_from_pdf(path)
        if not pdf_texts:
            raise Exception(f"Error: Could not extract text from {path}")

        # ── Screen: reject papers below score threshold ──────────────────────────
        _existing_screen = resolve_pipeline_output(effective_paper_id, "pipeline1/screen_paper")
        if _existing_screen:
            print(f"\n  [SKIP] Screen — loading existing: {os.path.basename(_existing_screen)}")
            with open(_existing_screen, "r", encoding="utf-8") as f:
                screen_result = json.load(f)
        else:
            screen_response = pipeline.screen_paper(pdf_texts)
            if not screen_response:
                raise Exception("Error: No response from screen LLM")
            screen_result = extract_json_from_response(screen_response)
            save_json_output(screen_result, "pipeline1/screen_paper", effective_paper_id)

        _screen_dims = ["student_behavior", "learning_context", "teacher_actions"]
        _failed_dims = [d for d in _screen_dims if screen_result.get(d, {}).get("score", 5) < 3]
        if _failed_dims:
            print(f"  [SCREEN REJECTED] {effective_paper_id}: {', '.join(_failed_dims)} < 3.")
            _existing_rejected = resolve_pipeline_output(effective_paper_id, "pipeline1/analyze_paper")
            if not _existing_rejected:
                _rejected_data = {"id": effective_paper_id, "screen": screen_result, "category_list": []}
                save_json_output(_rejected_data, "pipeline1/analyze_paper", effective_paper_id)
            return
        print(f"  [SCREEN PASSED] {effective_paper_id}: all required dimensions >= 3.")
        
        _existing_p1 = resolve_pipeline_output(effective_paper_id, "pipeline1/analyze_paper")
        if _existing_p1:
            print(f"\n  [SKIP] Pipeline 1 — loading existing: {os.path.basename(_existing_p1)}")
            with open(_existing_p1, "r", encoding="utf-8") as f:
                pipeline1_paper_data = json.load(f)
        else:
            analyze_paper_response = pipeline.analyze_paper(pdf_texts, paper_id=effective_paper_id)
            if not analyze_paper_response:
                raise Exception("Error: No response from LLM")

            pipeline1_paper_data = extract_json_from_response(analyze_paper_response)

            if args.verbose:
                print("\n=== Pipeline 1: Analyze Paper ===")
                print_pipeline1_verbose(pipeline1_paper_data)

            # Embed screen result + mark each category as eligible or not
            pipeline1_paper_data["screen"] = screen_result
            _p1_dims = ["student_behavior", "learning_context", "teacher_actions"]
            for _cat in pipeline1_paper_data.get("category_list", []):
                if not isinstance(_cat, dict):
                    continue
                _elig = _cat.get("extraction_eligibility", {})
                if not isinstance(_elig, dict):
                    _elig = {}
                def _score(v):
                    try: return int(v)
                    except (ValueError, TypeError): return 5
                _bad = [d for d in _p1_dims if _score(_elig.get(d, {}).get("score", 5)) <= 3]
                _cat["eligible"] = len(_bad) == 0
                if _bad:
                    print(f"  [CAT FILTERED] {_cat.get('category_id', '?')}: {', '.join(_bad)} <= 3.")

            json_path = save_json_output(pipeline1_paper_data, "pipeline1/analyze_paper", effective_paper_id)
            print(f">>> JSON saved: {json_path}")

        # For pipeline 2+: only pass eligible categories downstream
        _kept_cats = [c for c in pipeline1_paper_data.get("category_list", []) if c.get("eligible", True)]
        if not _kept_cats:
            print(f"  [PAPER SKIPPED] {effective_paper_id}: no eligible categories.")
            return
        pipeline1_paper_data = {**pipeline1_paper_data, "category_list": _kept_cats}

        logger = PipelineLogger(get_paper_id(pipeline1_paper_data, path))

        log_pipeline1_scores(pipeline1_paper_data, logger)

        if args.pipeline1_only:
            return

        if use_check:
            if input("Continue to Pipeline 2 (Extract Excerpt)? (y/n): ").strip().lower() != 'y':
                print("Exiting.")
                return


    # ── Pipeline 2: Extract Excerpt ───────────────────────────────────────────
    if hasattr(args, 'pipeline2_output_learner') and args.pipeline2_output_learner and \
       hasattr(args, 'pipeline2_output_instruction') and args.pipeline2_output_instruction:
        print(f">>> Loading pipeline 2 outputs")
        try:
            with open(args.pipeline2_output_learner, "r", encoding="utf-8") as f:
                pipeline2_excerpt_data.append(json.load(f))
            with open(args.pipeline2_output_instruction, "r", encoding="utf-8") as f:
                pipeline2_excerpt_data.append(json.load(f))
            print(f">>> Pipeline 2 loaded")
        except Exception as e:
            print(f"Error loading pipeline 2 output: {e}")
            return
    else:
        _paper_id_p2 = effective_paper_id or \
                       (pipeline1_paper_data.get("id") if pipeline1_paper_data else None) or \
                       (path.replace("\\", "/").split("/")[-1].replace(".pdf", "") if path else "unknown")
        _existing_p2_l = resolve_pipeline_output(_paper_id_p2, "pipeline2/extract_excerpt_learner")
        _existing_p2_i = resolve_pipeline_output(_paper_id_p2, "pipeline2/extract_excerpt_instruction")

        if _existing_p2_l and _existing_p2_i:
            print(f"\n  [SKIP] Pipeline 2 — loading existing for '{_paper_id_p2}'")
            with open(_existing_p2_l, "r", encoding="utf-8") as f:
                output_data1 = json.load(f)
            with open(_existing_p2_i, "r", encoding="utf-8") as f:
                output_data2 = json.load(f)
        else:
            extract_excerpt_response_learner, extract_excerpt_response_instruction = pipeline.extract_excerpt(
                path=args.path if not args.pipeline1_output else path,
                data=pipeline1_paper_data.get("category_list", []),
                excluded_pages=pipeline1_paper_data.get("irrelevant_pages", {}).get("page_numbers", []),
                paper_id=_paper_id_p2,
            )

            if not extract_excerpt_response_learner or not extract_excerpt_response_instruction:
                raise Exception("Error: No response from LLM")

            output_data1 = extract_json_from_response(extract_excerpt_response_learner)
            output_data2 = extract_json_from_response(extract_excerpt_response_instruction)

            if args.verbose:
                print("\n=== Pipeline 2: Extract Excerpt ===")
                print_pipeline2_verbose(output_data1, output_data2, pipeline1_paper_data)

            json_path = save_json_output(output_data1, "pipeline2/extract_excerpt_learner", _paper_id_p2)
            print(f">>> JSON saved: {json_path}")

            json_path = save_json_output(output_data2, "pipeline2/extract_excerpt_instruction", _paper_id_p2)
            print(f">>> JSON saved: {json_path}")

            if logger is None:
                logger = PipelineLogger(get_paper_id(pipeline1_paper_data, path))

        annotate_pipeline2(output_data1, output_data2, pipeline1_paper_data.get("category_list", []), logger)
        _p2_kept = [c for c in pipeline1_paper_data.get("category_list", [])
                    if c.get("quality_gate", {}).get("pipeline2_passed", False)]

        if not _p2_kept:
            print(f"  [P2 SKIPPED] {_paper_id_p2}: no categories with excerpts.")
            return
        pipeline1_paper_data = {**pipeline1_paper_data, "category_list": _p2_kept}

        pipeline2_excerpt_data.append(output_data1)
        pipeline2_excerpt_data.append(output_data2)

        if logger is None:
            logger = PipelineLogger(get_paper_id(pipeline1_paper_data, path))

        if args.pipeline2_only:
            return

        if use_check:
            if input("Continue to Pipeline 3 (Build Scenario)? (y/n): ").strip().lower() != 'y':
                print("Exiting.")
                return


    # ── Pipeline 3: Build Scenario ────────────────────────────────────────────
    if hasattr(args, 'pipeline3_output') and args.pipeline3_output:
        print(f">>> Loading {len(args.pipeline3_output)} pipeline 3 output(s)")
        for arg_pipeline3_output in args.pipeline3_output:
            try:
                with open(arg_pipeline3_output, "r", encoding="utf-8") as f:
                    pipeline3_scenario_data.append(json.load(f))
                print(f">>> Loaded: {os.path.basename(arg_pipeline3_output)}")
            except Exception as e:
                print(f"Error loading pipeline 3 output: {e}")
                return
    else:
        for category in pipeline1_paper_data.get("category_list", []):
            _cat_id_p3 = category["category_id"]
            _existing_p3 = resolve_pipeline_output(_cat_id_p3, "pipeline3/build_scenario")
            if _existing_p3:
                print(f"\n  [SKIP] Pipeline 3 [{_cat_id_p3}] — loading existing: {os.path.basename(_existing_p3)}")
                with open(_existing_p3, "r", encoding="utf-8") as f:
                    pipeline3_scenario_data.append(json.load(f))
                continue

            build_scenario_response = pipeline.build_scenario(
                paper_id=pipeline1_paper_data["id"],
                category=category,
                learner_excerpts=pipeline2_excerpt_data[0],
                instruction_excerpts=pipeline2_excerpt_data[1],
                num_scenarios=num_scenarios,
            )
            if not build_scenario_response:
                raise Exception("Error: No response from LLM")
            output_data = extract_json_from_response(build_scenario_response)
            if not output_data.get("category_id"):
                print(f"  [FAILED] Pipeline 3 [{_cat_id_p3}]: invalid JSON structure, skipping.")
                continue

            if args.verbose:
                print(f"\n=== Pipeline 3: Build Scenario ({category['category_id']}) ===")
                print_pipeline3_verbose(output_data)

            for _s in output_data.get("scenarios", []):
                _hs = _s.get("hidden_strategies", {})
                _s["is_actionable"] = bool(
                    _hs.get("effective_immediate") or _hs.get("effective_long_term")
                )

            json_path = save_json_output(output_data, "pipeline3/build_scenario", category["category_id"])
            print(f">>> JSON saved: {json_path}")

            if logger:
                logger.log_pipeline3(output_data)

            pipeline3_scenario_data.append(output_data)

        if args.pipeline3_only:
            return

        if use_check:
            if input("Continue to Pipeline 4 (Synthesize Dialogue)? (y/n): ").strip().lower() != 'y':
                print("Exiting.")
                return


    # ── Pipeline 4: Synthesize Dialogue ────────────────────────────────────────────
    # Build excerpt lookup from P2 outputs (excerpt_id → excerpt_text)
    _paper_id_p4 = pipeline1_paper_data["id"]
    _evidence_lookup: dict[str, str] = {}
    for _p2_learner_file in sorted(glob(os.path.join(os.getcwd(), "data", "pipeline2", "extract_excerpt_learner", f"extract_excerpt_learner_{_paper_id_p4}_*.json")))[-1:]:
        with open(_p2_learner_file, encoding="utf-8") as _f:
            _p2_data = json.load(_f)
            for _ex in _p2_data.get("learner_excerpts", []):
                _evidence_lookup[_ex["excerpt_id"]] = _ex["excerpt_text"]
            for _q in _p2_data.get("student_quotes", []):
                _evidence_lookup[_q["quote_id"]] = _q["quote_text"]

    def _collect_evidence_ids(obj) -> list[str]:
        ids = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "evidence_basis" and isinstance(v, list):
                    ids.extend(v)
                else:
                    ids.extend(_collect_evidence_ids(v))
        elif isinstance(obj, list):
            for item in obj:
                ids.extend(_collect_evidence_ids(item))
        return ids

    def _strip_evidence_basis(obj):
        if isinstance(obj, dict):
            return {k: _strip_evidence_basis(v) for k, v in obj.items() if k != "evidence_basis"}
        elif isinstance(obj, list):
            return [_strip_evidence_basis(item) for item in obj]
        return obj

    if hasattr(args, 'pipeline4_output') and args.pipeline4_output:
        print(f">>> Loading {len(args.pipeline4_output)} pipeline 4 output(s)")
        for arg_pipeline4_output in args.pipeline4_output:
            try:
                with open(arg_pipeline4_output, "r", encoding="utf-8") as f:
                    pipeline4_dialogue_data.append(json.load(f))
                print(f">>> Loaded: {os.path.basename(arg_pipeline4_output)}")
            except Exception as e:
                print(f"Error loading pipeline 4 output: {e}")
                return
    else:
        for category, scenario_data in zip(pipeline1_paper_data.get("category_list", []), pipeline3_scenario_data):
            individual_scenarios = scenario_data.get("scenarios", [])
            if not individual_scenarios:
                print(f"Warning: No scenarios found for {category.get('category_id')}")
                continue

            _cat_id_p4 = category.get("category_id")
            _p4_dir = os.path.join(os.getcwd(), "data", "pipeline4", "synthesize_dialogue")

            # Load existing P4 outputs indexed by source_scenario_id
            _existing_p4_files = sorted(glob(os.path.join(_p4_dir, f"synthesize_dialogue_{_cat_id_p4}_*.json")))
            _done_scenarios = {}
            _max_case_num = 0
            for _p4_path in _existing_p4_files:
                with open(_p4_path, "r", encoding="utf-8") as f:
                    _p4_data = json.load(f)
                for _dlg in _p4_data.get("dialogues", []):
                    _src = _dlg.get("source_scenario_id")
                    if _src:
                        _done_scenarios[_src] = _p4_data
                    _case_match = re.search(r"_case_(\d+)", _dlg.get("case_id", ""))
                    if _case_match:
                        _max_case_num = max(_max_case_num, int(_case_match.group(1)))

            case_counter = _max_case_num

            for individual_scenario in individual_scenarios:
                scenario_id = individual_scenario.get("scenario_id", "unknown")
                if not individual_scenario.get("is_actionable", True):
                    print(f"\n  [SKIP] Scenario {scenario_id} — no effective strategies (is_actionable=false)")
                    continue

                if scenario_id in _done_scenarios:
                    print(f"\n  [SKIP] Pipeline 4 [{scenario_id}] — loading existing")
                    pipeline4_dialogue_data.append(_done_scenarios[scenario_id])
                    continue

                print(f"\n>>> Processing scenario: {scenario_id}")

                turn = random.randint(min_turn, max_turn)
                print(f"    TURN: {turn} (dialogue turns: {turn * 2})")

                scenario_visible = {
                    "scenario_id": individual_scenario.get("scenario_id"),
                    "title": individual_scenario.get("title"),
                    "content": individual_scenario.get("content"),
                }

                # Resolve evidence_basis IDs → actual excerpt texts (scenario content only)
                _evidence_ids = _collect_evidence_ids(individual_scenario.get("content", {}))
                evidence = {eid: _evidence_lookup[eid] for eid in dict.fromkeys(_evidence_ids) if eid in _evidence_lookup}

                synthesize_dialogue_response = pipeline.synthesize_dialogue(
                    paper_id=pipeline1_paper_data["id"],
                    category=category,
                    scenario_data=_strip_evidence_basis(scenario_data),
                    individual_scenario=individual_scenario,
                    evidence=evidence,
                    turn=turn,
                )
                if not synthesize_dialogue_response:
                    raise Exception("Error: No response from LLM")

                output_data = extract_json_from_response(synthesize_dialogue_response)

                # Renumber case_ids and inject scenario_visible as full_context
                _cat_id = category.get("category_id", "unknown")
                for dlg in output_data.get("dialogues", []):
                    case_counter += 1
                    dlg["case_id"] = f"{_cat_id}_case_{case_counter:03d}"
                    dlg["source_scenario_id"] = scenario_id
                    dlg["full_context"] = scenario_visible

                if args.verbose:
                    print(f"\n=== Pipeline 4: Synthesize Dialogue ({category['category_id']}) ===")
                    print_pipeline4_verbose(output_data)

                json_path = save_json_output(output_data, "pipeline4/synthesize_dialogue", category["category_id"])
                print(f">>> JSON saved: {json_path}")

                print(f">>> Generating dialogue files...")
                try:
                    category_id = output_data.get("category_id")
                    _hs = individual_scenario.get("hidden_strategies", {})
                    _has_immediate = bool(_hs.get("effective_immediate"))
                    _has_long_term = bool(_hs.get("effective_long_term"))
                    created_files = generate_all_dialogue_files(
                        category_id, output_data,
                        has_immediate=_has_immediate,
                        has_long_term=_has_long_term,
                    )
                    print(f">>> Generated {len(created_files)} dialogue files for {category_id}")
                except Exception as e:
                    print(f"Error generating dialogue files: {e}")

                if logger is None:
                    logger = PipelineLogger(get_paper_id(pipeline1_paper_data))

                logger.log_pipeline4(output_data)

                pipeline4_dialogue_data.append(output_data)

        if args.pipeline4_only:
            print(">>> --pipeline4-only: stopping after Pipeline 4.")
            return

        if use_check:
            if input("\nContinue to Pipeline 5 (Generate Responses)? (y/n): ").strip().lower() != 'y':
                print("Exiting.")
                return


    # ── Pipeline 5: Generate Responses ────────────────────────────────────────────
    if hasattr(args, 'pipeline5_dialogue_files') and args.pipeline5_dialogue_files:
        print(f"\n>>> Pipeline 5: {len(args.pipeline5_dialogue_files)} dialogue file(s)")

        category_ids = set()
        for dialogue_file in args.pipeline5_dialogue_files:
            filename = os.path.basename(dialogue_file)
            # Accept both bare IDs (0001-01) and namespaced IDs (adhd_0001-01).
            match = re.match(r'(?P<category_id>.+?)_case_', filename)
            if match:
                category_ids.add(match.group("category_id"))

        print(f">>> Categories: {', '.join(sorted(category_ids))}")

        for category_id in sorted(category_ids):
            print(f"\n>>> Collecting responses for: {category_id}")
            try:
                responses = collect_model_responses(category_id, args.response_config)
                print(f">>> Collected {len(responses)} responses")
            except Exception as e:
                print(f"Error collecting responses: {e}")
                continue
    else:
        seen_cat_ids = set()
        for dialogue_data in pipeline4_dialogue_data:
            category_id = dialogue_data.get("category_id")
            if category_id and category_id not in seen_cat_ids:
                seen_cat_ids.add(category_id)
                print(f"\n>>> Collecting responses for: {category_id}")
                try:
                    responses = collect_model_responses(category_id, args.response_config)
                    print(f">>> Collected {len(responses)} responses")
                except Exception as e:
                    print(f"Error collecting responses: {e}")
                    continue

    if args.pipeline5_only:
        print(">>> --pipeline5-only: stopping after Pipeline 5.")
        return


if __name__ == "__main__":
    main()