import os
import re


# =============================================================================
# utils/generate_dialogue_utils.py — Dialogue .txt File Generator
# =============================================================================
#
# Generates immediate and long_term evaluation .txt files from
# synthesize_dialogue output. Called at the end of Pipeline 4.
#
# Output: data/pipeline4/dialogue/{category_id}_case_{n}_{type}.txt
#
# =============================================================================


def format_dialogue_base(category_name: str, category_id: str, case_id: str,
                        minimal_context: str, dialogue_turns: list) -> str:
    """Format the base dialogue section (same for both files)"""
    separator = "─" * 46
    
    content = f"""{'='*50}
CATEGORY ID: {category_id}
CASE ID: {case_id}
{'='*50}

{separator}
SCENARIO
{separator}

{minimal_context}

{separator}
DIALOGUE
{separator}

"""
    
    for turn in dialogue_turns:
        speaker = turn.get("speaker", "unknown").upper()
        message = turn.get("message", "")
        content += f"{speaker}: {message}\n\n"
    
    return content


def format_immediate_response_file(category_name: str, category_id: str, case_id: str,
                                   minimal_context: str, dialogue_turns: list) -> str:
    """Format the immediate response file"""
    separator = "─" * 46
    
    base = format_dialogue_base(category_name, category_id, case_id, minimal_context, dialogue_turns)
    
    prompt = f"""{separator}
YOUR TURN - IMMEDIATE RESPONSE
{separator}

The student has just exhibited challenging behavior.

What would you say or do RIGHT NOW as the tutor in this moment?

Provide a direct, immediate response (2-3 sentences) in this moment. Do NOT provide a multi-step plan or describe what you will do later.

Your immediate response (Output Response ONLY): 
"""
    
    return base + prompt


def format_long_term_intervention_file(category_name: str, category_id: str, case_id: str,
                                       minimal_context: str, dialogue_turns: list) -> str:
    """Format the long-term intervention file"""
    separator = "─" * 46
    
    base = format_dialogue_base(category_name, category_id, case_id, minimal_context, dialogue_turns)
    
    prompt = f"""{separator}
YOUR TURN - LONG-TERM INTERVENTION
{separator}

The student has just exhibited challenging behavior.

What sustained interventions should be implemented over time for this student?

Describe what long-term strategies and support structures you would put in place (2-3 sentences). This should be a narrative description of ongoing interventions over weeks or months, not immediate actions.

Your long-term intervention plan (Output Intervention Plan ONLY): 
"""
    
    return base + prompt


def generate_all_dialogue_files(category_id: str, dialogue_data: dict, output_dir: str = None,
                                has_immediate: bool = True, has_long_term: bool = True):
    """
    Generate dialogue text files (TWO per case: immediate and long-term)
    
    Args:
        category_id: Category ID (e.g., "0004-01")
        dialogue_data: Dialogue data dict from synthesize_dialogue
        output_dir: Output directory (default: ./dialogue/)
    
    Returns:
        List of created file paths
    """
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "data", "pipeline4", "dialogue")
    
    os.makedirs(output_dir, exist_ok=True)
    
    category_name = dialogue_data.get("name", "Unknown Category")
    dialogues = dialogue_data.get("dialogues", [])
    
    if not dialogues:
        print("Warning: No dialogues found in data")
        return []
    
    created_files = []
    
    for idx, dialogue_case in enumerate(dialogues, 1):
        case_id = dialogue_case.get("case_id", "unknown")
        minimal_context = dialogue_case.get("minimal_context", "No context provided")
        dialogue_turns = dialogue_case.get("dialogue", [])
        
        if not dialogue_turns:
            print(f"Warning: No dialogue turns for case {case_id}")
            continue
        
        case_match = re.search(r'_case_(\d+)', case_id)
        case_num = case_match.group(1) if case_match else f"{idx:03d}"
        
        if has_immediate:
            immediate_filename = f"{category_id}_case_{case_num}_immediate.txt"
            immediate_filepath = os.path.join(output_dir, immediate_filename)
            immediate_content = format_immediate_response_file(
                category_name, category_id, case_id,
                minimal_context, dialogue_turns
            )
            try:
                with open(immediate_filepath, "w", encoding="utf-8") as f:
                    f.write(immediate_content)
                print(f"  Created: {immediate_filename}")
                created_files.append(immediate_filepath)
            except Exception as e:
                print(f"  Error creating {immediate_filename}: {e}")

        if has_long_term:
            long_term_filename = f"{category_id}_case_{case_num}_long_term.txt"
            long_term_filepath = os.path.join(output_dir, long_term_filename)
            long_term_content = format_long_term_intervention_file(
                category_name, category_id, case_id,
                minimal_context, dialogue_turns
            )
            try:
                with open(long_term_filepath, "w", encoding="utf-8") as f:
                    f.write(long_term_content)
                print(f"  Created: {long_term_filename}")
                created_files.append(long_term_filepath)
            except Exception as e:
                print(f"  Error creating {long_term_filename}: {e}")
    
    return created_files