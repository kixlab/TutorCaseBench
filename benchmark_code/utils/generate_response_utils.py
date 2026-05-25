import os
import json
import glob
from dotenv import load_dotenv
from utils.llm_utils import generate_llm_response


# =============================================================================
# utils/generate_response_utils.py — Model Response Collector
# =============================================================================
#
# Collects responses from all models in config_response.json
# for each dialogue file in data/pipeline4/dialogue/. Called in Pipeline 5.
#
# Skips if response file already exists.
# Output: data/pipeline5/response/{cat_id}_case_{n}_{type}_{model}.txt
#
# =============================================================================


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(_PROJECT_ROOT, ".env.local"))


def safe_model_label(model_name: str) -> str:
    """Filename-safe alias for a model id.
    Together IDs contain '/', Bedrock IDs contain ':' — both unsafe in filenames.
    Apply this on both write (response filename) and read (matching models_to_test
    against existing response files) so the two stay symmetric.
    """
    return model_name.replace("/", "-").replace(":", "-")


def load_config(config_path: str = "config/config_response.json"):
    """Load configuration file"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None


def collect_model_responses(category_id: str, config_path: str = "config/config_response.json"):
    """
    Collect responses from configured models for all dialogue files.

    Args:
        category_id: Category ID (e.g., "0004-01")
        config_path: Path to config file

    Returns:
        List of response info dicts
    """
    config = load_config(config_path)
    if not config:
        return []

    models = config.get("models_to_test", [])
    dialogue_dir = config.get("output_dirs", {}).get("dialogue", os.path.join("data", "pipeline4", "dialogue"))
    response_dir = config.get("output_dirs", {}).get("response", os.path.join("data", "pipeline5", "response"))

    os.makedirs(response_dir, exist_ok=True)

    immediate_pattern = os.path.join(dialogue_dir, f"{category_id}_case_*_immediate.txt")
    long_term_pattern = os.path.join(dialogue_dir, f"{category_id}_case_*_long_term.txt")
    dialogue_files = glob.glob(immediate_pattern) + glob.glob(long_term_pattern)

    if not dialogue_files:
        print(f"No dialogue files found for category {category_id}")
        return []

    print(f">>> Found {len(dialogue_files)} dialogue files")

    all_responses = []

    for dialogue_path in dialogue_files:
        filename = os.path.basename(dialogue_path).replace(".txt", "")

        if "_immediate.txt" in dialogue_path:
            response_type = "immediate"
        elif "_long_term.txt" in dialogue_path:
            response_type = "long_term"
        else:
            continue

        print(f"\n  Processing: {filename}")

        try:
            with open(dialogue_path, "r", encoding="utf-8") as f:
                dialogue_content = f.read()
        except Exception as e:
            print(f"    Error reading file: {e}")
            continue

        for model_config in models:
            model_name = model_config["name"]       # e.g. "gpt-4o", "gemini-2.0-flash"
            model_type = model_config["model_type"] # e.g. "openai", "gemini"

            response_filename = f"{filename}_{safe_model_label(model_name)}.txt"
            response_path = os.path.join(response_dir, response_filename)

            if os.path.exists(response_path):
                print(f"    (v) Response exists: {response_filename}")
                all_responses.append({
                    "dialogue_file": dialogue_path,
                    "response_type": response_type,
                    "model_name": model_name,
                    "response_path": response_path,
                    "status": "existing"
                })
                continue

            print(f"    Generating response: {model_name} ({model_type})")
            try:
                response = generate_llm_response(
                    prompt_filename="generate_response",
                    variables={"PROMPT": dialogue_content},
                    model_name=model_name,
                    model_type=model_type,
                    use_api_defaults=True,
                )

                if not response:
                    print(f"    (x) No response from {model_name}")
                    continue

                # Some models (notably gpt-oss-120b via Together) occasionally emit
                # a trailing NULL byte that makes text editors treat the file as binary.
                cleaned = response.strip().replace("\x00", "")
                with open(response_path, "w", encoding="utf-8") as f:
                    f.write(cleaned)

                print(f"    (v) Saved: {response_filename}")

                all_responses.append({
                    "dialogue_file": dialogue_path,
                    "response_type": response_type,
                    "model_name": model_name,
                    "response_path": response_path,
                    "response_text": cleaned,
                    "status": "new"
                })

            except Exception as e:
                print(f"    (x) Error generating response: {e}")
                continue

    print(f"\n>>> Collected {len(all_responses)} total responses")
    return all_responses