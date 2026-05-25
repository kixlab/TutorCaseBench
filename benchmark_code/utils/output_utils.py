import os
import json
from datetime import datetime
from typing import Dict, Any


# =============================================================================
# utils/output_utils.py — JSON / Text File Saver
# =============================================================================
#
# Saves pipeline outputs to data/{pipeline_name}/ with timestamp-based filenames.
# Output format: {step_name}_{id}_{ts}.json
#
# =============================================================================


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def save_llm_response(response_text: str, pipeline_name: str, filename: str = None) -> str:
    """
    Save raw LLM response text to a file in the output/ folder.
    
    Creates output/{pipeline_name}/ directory if it doesn't exist and saves the response there.
    If no filename is provided, generates one with timestamp.
    
    Args:
        response_text: Raw text response from LLM
        pipeline_name: Name of the pipeline (e.g., 'paper_analysis', 'student_extraction')
        filename: Optional filename (without .txt extension). If None, generates timestamp-based name.
        
    Returns:
        str: Full path to the saved file
        
    Raises:
        IOError: If the file cannot be written
    """
    step_name = os.path.basename(pipeline_name)
    output_dir = os.path.join(_PROJECT_ROOT, "data", pipeline_name, "raw")
    os.makedirs(output_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{step_name}_response_{timestamp}"

    if not filename.endswith(".txt"):
        filename += ".txt"
    
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(response_text)
        return filepath
    except IOError as e:
        raise IOError(f"Failed to save response to {filepath}: {e}")


def save_json_output(data: Dict[str, Any], pipeline_name: str, filename: str = None) -> str:
    """
    Save JSON data to a file in the output/ folder with pipeline-specific subdirectories.
    
    Creates output/{pipeline_name}/ directory if it doesn't exist and saves the JSON file there.
    If no filename is provided, generates one with timestamp.
    
    Args:
        data: Dictionary to save as JSON
        pipeline_name: Name of the pipeline (e.g., 'paper_analysis', 'student_extraction')
        filename: Optional filename (without .json extension). If None, generates timestamp-based name.
        
    Returns:
        str: Full path to the saved file
        
    Raises:
        IOError: If the file cannot be written
    """
    step_name = os.path.basename(pipeline_name)
    output_dir = os.path.join(_PROJECT_ROOT, "data", pipeline_name)
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{step_name}_{timestamp}"
    else:
        filename = filename.strip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{step_name}_{filename}_{timestamp}"
    
    if not filename.endswith(".json"):
        filename += ".json"
    
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        return filepath
    except IOError as e:
        raise IOError(f"Failed to save JSON to {filepath}: {e}")