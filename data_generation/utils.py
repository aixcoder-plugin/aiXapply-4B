"""
aiXapply data generation pipeline.

Stage: shared utilities.
Purpose: provide language mappings, jsonl writing, cost estimates, and merge helpers.
Inputs: request records and per-sample dictionaries.
Outputs: files and normalized in-memory records.
"""
import json
from pathlib import Path

# Language extension name mapping
EXTENSION_MAP: dict[str, str] = {
    "python": "py",
    "javascript": "js",
    "java": "java",
    "c++": "cpp",
    "c": "c",
    "go": "go",
    "typescript": "ts",
    "csharp": "cs",
    "rust": "rs",
    "json": "json",
    "yaml": "yaml",
    "xml": "xml",
    "sql": "sql",
    "ini": "ini",
    "markdown": "md",
    "text": "txt",
    "html": "html",
    "restructuredtext": "rst",
    "shell": "sh",
    "dockerfile": "dockerfile",
    "makefile": "makefile",
}

def write_jsonl(requests: list, output_dir: str, batch_number: int = 1):
    """
    Writes the list of requests to a .jsonl file.
    
    Args:
        requests (list): List of request dictionaries.
        output_dir (str): Path to the output directory.
        batch_number (int): Batch number to append to the filename.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"batch_{batch_number:03d}.jsonl"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + '\n')
    print(f"Batch input file created at {output_file}")

def calculate_cost(input_tokens, output_tokens):
    input_cost = (input_tokens / 1_000_000) * 1.25  # $1.25 per 1M tokens for input, for DeepSeek API
    output_cost = (output_tokens / 1_000_000) * 5  # $5 per 1M tokens for output, for DeepSeek API
    return input_cost + output_cost


def update_data_pair(data_pair, custom_id_in, data_info: dict):
    if custom_id_in not in data_pair:
        data_pair[custom_id_in] = data_info
    else:
        for key, value in data_info.items():
            data_pair[custom_id_in][key] = value
    return data_pair