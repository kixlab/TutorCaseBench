import os
import re
from glob import glob


# =============================================================================
# utils/pipeline_utils.py — Dependency Auto-Resolver & Path Lookup
# =============================================================================
#
# Parses paper_id / category_id from --pipelineN-output arguments and
# auto-resolves upstream outputs so later pipeline stages can be resumed
# without manually specifying every intermediate file path.
#
# =============================================================================


def resolve_pipeline_output(short_id: str, pipeline_name: str, pattern_suffix: str = None):
    """
    Resolve short ID to full filepath.
    
    Args:
        short_id: Short identifier (e.g., "0001", "0001-01")
        pipeline_name: Pipeline directory name (e.g., "analyze_paper", "build_scenario")
        pattern_suffix: Optional suffix for pattern (e.g., "_*" for wildcard)
    
    Returns:
        Full filepath or None
    """
    step_name = os.path.basename(pipeline_name)
    search_dir = os.path.join(os.getcwd(), "data", pipeline_name)

    if pattern_suffix is None:
        pattern_suffix = "_*"

    pattern = os.path.join(search_dir, f"{step_name}_{short_id}{pattern_suffix}.json")
    matches = glob(pattern)
    
    if matches:
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return matches[0]
    return None


def find_pdf_path(paper_id: str):
    """
    Find PDF file path for a given paper ID.
    Accepts both plain IDs ("0001") and study-group-prefixed IDs ("adhd_0001").
    Searches current directory, docs/, and one level of subdirectories under docs/.

    Args:
        paper_id: Paper identifier (e.g., "0001" or "adhd_0001")

    Returns:
        PDF filepath or None
    """
    docs_dir = os.path.join(os.getcwd(), "data", "docs")

    # If paper_id is prefixed with a study group (e.g. "adhd_0001"),
    # try docs/{study_group}/{bare_id}.pdf first.
    parts = paper_id.rsplit("_", 1)
    if len(parts) == 2 and re.match(r"^\d+$", parts[1]):
        study_group, bare_id = parts
        candidate = os.path.join(docs_dir, study_group, f"{bare_id}.pdf")
        if os.path.exists(candidate):
            return candidate

    search_dirs = [os.getcwd()]
    if os.path.isdir(docs_dir):
        search_dirs.append(docs_dir)
        for entry in os.scandir(docs_dir):
            if entry.is_dir() and not entry.name.startswith("_"):
                search_dirs.append(entry.path)

    for search_dir in search_dirs:
        candidate = os.path.join(search_dir, f"{paper_id}.pdf")
        if os.path.exists(candidate):
            return candidate

    return None


def auto_resolve_dependencies(args):
    """
    Automatically resolve pipeline dependencies.
    Fills in earlier pipeline outputs based on later pipeline inputs.
    """
    starting_pipeline = None
    paper_id = None
    category_id = None
    case_id = None
    response_type = None

    # Pipeline 5: Generate Dialogue (--pipeline5-output)
    if hasattr(args, 'pipeline5_output') and args.pipeline5_output:
        starting_pipeline = 5

        # Handles both plain IDs ("0001-01_case_001_immediate") and
        # study-group-prefixed IDs ("adhd_0001-01_case_001_immediate")
        match = re.match(r'^(.+)-(\d{2})_case_(\d{3})_(immediate|long_term)', args.pipeline5_output)
        if match:
            paper_id = match.group(1)
            category_id = f"{match.group(1)}-{match.group(2)}"
            case_id = match.group(3)
            response_type = match.group(4)
        else:
            match = re.match(r'^(.+)-(\d{2})_case_(\d{3})', args.pipeline5_output)
            if match:
                paper_id = match.group(1)
                category_id = f"{match.group(1)}-{match.group(2)}"
                case_id = match.group(3)
            else:
                match = re.match(r'^(.+)-(\d{2})', args.pipeline5_output)
                if match:
                    paper_id = match.group(1)
                    category_id = f"{match.group(1)}-{match.group(2)}"
                else:
                    paper_id = args.pipeline5_output

        print(f">>> Pipeline5 mode: paper={paper_id}, category={category_id}, case={case_id}, type={response_type}")

        dialogue_dir = os.path.join(os.getcwd(), "data", "pipeline4", "dialogue")

        if paper_id and category_id and case_id and response_type:
            pattern = f"{category_id}_case_{case_id}_{response_type}.txt"
        elif paper_id and category_id and case_id:
            pattern = f"{category_id}_case_{case_id}_*.txt"
        elif paper_id and category_id:
            pattern = f"{category_id}_case_*.txt"
        elif paper_id:
            pattern = f"{paper_id}-*_case_*.txt"
        else:
            print(">>> Error: Invalid pipeline5-output format")
            return False

        dialogue_files = glob(os.path.join(dialogue_dir, pattern))

        if not dialogue_files:
            print(f">>> Error: No dialogue files found matching pattern: {pattern}")
            return False

        dialogue_files.sort()
        print(f">>> Found {len(dialogue_files)} dialogue files")
        args.pipeline5_dialogue_files = dialogue_files


    # Pipeline 4: Synthesize Dialogue (--pipeline4-output)
    elif hasattr(args, 'pipeline4_output') and args.pipeline4_output:
        starting_pipeline = 4

        match = re.match(r'^(.+)-(\d{2})', args.pipeline4_output)
        if match:
            paper_id = match.group(1)
            category_id = f"{match.group(1)}-{match.group(2)}"
        else:
            paper_id = args.pipeline4_output

        print(f">>> Pipeline4 mode: paper={paper_id}, category={category_id}")


    # Pipeline 3: Build Scenario (--pipeline3-output)
    elif hasattr(args, 'pipeline3_output') and args.pipeline3_output:
        starting_pipeline = 3

        match = re.match(r'^(.+)-(\d{2})', args.pipeline3_output)
        if match:
            paper_id = match.group(1)
            category_id = f"{match.group(1)}-{match.group(2)}"
        else:
            paper_id = args.pipeline3_output

        print(f">>> Pipeline3 mode: paper={paper_id}, category={category_id}")


    # Pipeline 2: Extract Excerpt (--pipeline2-output)
    elif hasattr(args, 'pipeline2_output') and args.pipeline2_output:
        starting_pipeline = 2
        paper_id = args.pipeline2_output
        print(f">>> Pipeline2 mode: paper={paper_id}")


    # Pipeline 1: Analyze Paper (--pipeline1-output)
    elif hasattr(args, 'pipeline1_output') and args.pipeline1_output:
        starting_pipeline = 1

        if not os.path.exists(args.pipeline1_output):
            # Treat as short paper ID (e.g. "0001" or "adhd_0001")
            paper_id = args.pipeline1_output
            print(f">>> Pipeline1 mode: paper={paper_id}")
        else:
            basename = os.path.basename(args.pipeline1_output)
            match = re.match(r'analyze_paper_(.+)_\d{8}_\d{6}\.json', basename)
            if match:
                paper_id = match.group(1)
                print(f">>> Pipeline1 mode (full path): paper={paper_id}")

    else:
        return True


    # Auto-fill pipeline1-output
    if paper_id and starting_pipeline and starting_pipeline >= 1:
        if hasattr(args, 'pipeline1_output') and args.pipeline1_output:
            if not os.path.exists(args.pipeline1_output):
                resolved = resolve_pipeline_output(paper_id, "pipeline1/analyze_paper")
                if resolved:
                    args.pipeline1_output = resolved
                    print(f">>> Auto-resolved pipeline1-output: {resolved}")
                else:
                    print(f">>> Warning: Could not resolve pipeline1-output for paper {paper_id}")
        else:
            resolved = resolve_pipeline_output(paper_id, "pipeline1/analyze_paper")
            if resolved:
                args.pipeline1_output = resolved
                print(f">>> Auto-resolved pipeline1-output: {resolved}")
            else:
                if starting_pipeline >= 2:
                    print(f">>> Warning: Could not resolve pipeline1-output for paper {paper_id}")

        if not hasattr(args, 'path') or not args.path:
            pdf_path = find_pdf_path(paper_id)
            if pdf_path:
                args.path = pdf_path
                print(f">>> Auto-resolved PDF path: {pdf_path}")


    # Auto-fill pipeline2-output
    if paper_id and starting_pipeline and starting_pipeline >= 2:
        if not (hasattr(args, 'pipeline2_output_learner') and args.pipeline2_output_learner):
            learner_resolved = resolve_pipeline_output(paper_id, "pipeline2/extract_excerpt_learner")
            instruction_resolved = resolve_pipeline_output(paper_id, "pipeline2/extract_excerpt_instruction")

            if learner_resolved and instruction_resolved:
                args.pipeline2_output_learner = learner_resolved
                args.pipeline2_output_instruction = instruction_resolved
                print(f">>> Auto-resolved pipeline2-output:")
                print(f"    Learner: {learner_resolved}")
                print(f"    Instruction: {instruction_resolved}")
            else:
                if starting_pipeline >= 3:
                    print(f">>> Warning: Could not resolve pipeline2-output for paper {paper_id}")


    # Auto-fill pipeline3-output (ALWAYS AS LIST)
    if starting_pipeline and starting_pipeline >= 3:
        resolved_scenarios = []

        if category_id:
            if hasattr(args, 'pipeline3_output') and args.pipeline3_output:
                if not os.path.exists(args.pipeline3_output):
                    resolved = resolve_pipeline_output(category_id, "pipeline3/build_scenario")
                    if resolved:
                        resolved_scenarios.append(resolved)
                    else:
                        if starting_pipeline >= 4:
                            print(f">>> Warning: Could not resolve pipeline3-output for category {category_id}")
                else:
                    resolved_scenarios.append(args.pipeline3_output)
            else:
                resolved = resolve_pipeline_output(category_id, "pipeline3/build_scenario")
                if resolved:
                    resolved_scenarios.append(resolved)
                else:
                    if starting_pipeline >= 4:
                        print(f">>> Warning: Could not resolve pipeline3-output for category {category_id}")
        elif paper_id:
            search_dir = os.path.join(os.getcwd(), "data", "pipeline3", "build_scenario")
            pattern = os.path.join(search_dir, f"build_scenario_{paper_id}-*_*.json")
            all_scenarios = glob(pattern)
            if all_scenarios:
                all_scenarios.sort()
                resolved_scenarios.extend(all_scenarios)
            else:
                if starting_pipeline >= 4:
                    print(f">>> Warning: Could not find any pipeline3 outputs for paper {paper_id}")

        if resolved_scenarios:
            args.pipeline3_output = resolved_scenarios
            print(f">>> Auto-resolved pipeline3-output: {len(resolved_scenarios)} file(s)")
            for scenario_path in resolved_scenarios:
                print(f"    - {os.path.basename(scenario_path)}")


    # Auto-fill pipeline4-output (ALWAYS AS LIST)
    if starting_pipeline and starting_pipeline >= 4:
        resolved_dialogues = []

        if category_id:
            if hasattr(args, 'pipeline4_output') and args.pipeline4_output:
                if not os.path.exists(args.pipeline4_output):
                    resolved = resolve_pipeline_output(category_id, "pipeline4/synthesize_dialogue")
                    if resolved:
                        resolved_dialogues.append(resolved)
                    else:
                        if starting_pipeline >= 5:
                            print(f">>> Warning: Could not resolve pipeline4-output for category {category_id}")
                else:
                    resolved_dialogues.append(args.pipeline4_output)
            else:
                resolved = resolve_pipeline_output(category_id, "pipeline4/synthesize_dialogue")
                if resolved:
                    resolved_dialogues.append(resolved)
                else:
                    if starting_pipeline >= 5:
                        print(f">>> Warning: Could not resolve pipeline4-output for category {category_id}")
        elif paper_id:
            search_dir = os.path.join(os.getcwd(), "data", "pipeline4", "synthesize_dialogue")
            pattern = os.path.join(search_dir, f"synthesize_dialogue_{paper_id}-*_*.json")
            all_dialogues = glob(pattern)
            if all_dialogues:
                all_dialogues.sort()
                resolved_dialogues.extend(all_dialogues)
            else:
                if starting_pipeline >= 5:
                    print(f">>> Warning: Could not find any pipeline4 outputs for paper {paper_id}")

        if resolved_dialogues:
            args.pipeline4_output = resolved_dialogues
            print(f">>> Auto-resolved pipeline4-output: {len(resolved_dialogues)} file(s)")
            for dialogue_path in resolved_dialogues:
                print(f"    - {os.path.basename(dialogue_path)}")

    print(f">>> Auto-resolve complete. Starting from pipeline {starting_pipeline}")
    return True