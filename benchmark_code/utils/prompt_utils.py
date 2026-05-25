import os
import re
import json
from typing import Dict
from json_repair import repair_json


# =============================================================================
# utils/prompt_utils.py — Prompt Builder & JSON Parser
# =============================================================================
#
# Loads prompts/{name}.txt and substitutes {KEY} placeholders.
# Parses JSON from LLM responses with json_repair fallback.
#
# =============================================================================


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def build_prompt(prompt_name: str, variables: Dict[str, str]) -> str:
    """
    Build a prompt by loading a template file and substituting variables.
    
    Args:
        prompt_name: Name of the prompt template file (e.g., 'student_extraction.txt')
        variables: Dictionary of placeholder names and their values to substitute in the template
        
    Returns:
        str: The complete prompt with all variables substituted
        
    Raises:
        FileNotFoundError: If the prompt file is not found in the prompts directory
    """
    prompts_dir = os.path.join(_PROJECT_ROOT, "prompts")

    if prompt_name.endswith('.txt'):
        prompt_path = os.path.join(prompts_dir, prompt_name)
    else:
        prompt_path = os.path.join(prompts_dir, prompt_name + ".txt")

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    for key, value in variables.items():
        template = template.replace(f"{{{key}}}", value)

    return template


def extract_json_from_response(text: str) -> Dict:
    """
    Extract and parse JSON from LLM response text.

    Strips code fences, attempts direct parsing, then falls back to
    brace-balanced extraction to correctly handle nested JSON objects.

    Args:
        text: Raw text response from the LLM

    Returns:
        Dict: Parsed JSON object

    Raises:
        json.JSONDecodeError: If no valid JSON is found in the text
    """
    text = text.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    def _try_parse(s: str) -> dict | None:
        """Try parsing JSON, with fallbacks for common LLM output issues."""
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # Fix invalid escape sequences (e.g. \' is not valid JSON)
        fixed = re.sub(r"\\'", "'", s)
        # Strip trailing commas before ] or }
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        # Last resort: use json_repair to handle malformed JSON
        # (e.g. unescaped quotes inside string values from verbatim student quotes)
        try:
            repaired = repair_json(s, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
        except Exception:
            pass
        return None

    # Try direct parse
    result = _try_parse(text)
    if result is not None:
        return result

    # Brace-balanced extraction: find the outermost { ... } block
    for i, ch in enumerate(text):
        if ch == '{':
            depth = 0
            for j in range(i, len(text)):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                if depth == 0:
                    candidate = text[i:j + 1]
                    result = _try_parse(candidate)
                    if result is not None:
                        return result
                    break  # this outer block didn't parse; keep scanning
            # If the balanced block didn't parse, continue to next '{'

    raise json.JSONDecodeError("No JSON object found in response", text, 0)