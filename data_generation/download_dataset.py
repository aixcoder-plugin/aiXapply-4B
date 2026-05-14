"""
aiXapply data generation pipeline.

Stage: source dataset download.
Purpose: download one CommitPack jsonl shard per configured language.
Inputs: sample_config.json and the Hugging Face CommitPack dataset.
Outputs: local CommitPack sample files under dataset/commitpack_samples.
"""
import json
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
# Try to import get_token method, compatible with different versions
try:
    from huggingface_hub import get_token
except ImportError:
    from huggingface_hub import HfFolder
    def get_token():
        return HfFolder.get_token()

CONFIG_PATH = Path(__file__).resolve().parent / "sample_config.json"
target_structure = json.load(open(CONFIG_PATH, "r", encoding="utf-8"))
target_structure = dict(target_structure)


REPO_ID = "bigcode/commitpack"
REPO_TYPE = "dataset"
LOCAL_DIR = "./dataset/commitpack_samples"

def list_files_via_api(repo_id, folder_path):
    """
    Get file list using Hugging Face HTTP API to avoid version-incompatible list_repo_files parameters.
    """
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    # Construct API URL (note: using datasets tree API)
    # Format: https://huggingface.co/api/datasets/{repo_id}/tree/main/{path}
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main/{folder_path}"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            # API returns a list, each element contains 'path' (full path)
            return [item['path'] for item in response.json() if item['type'] == 'file']
        else:
            print(f"  [API Warning] Unable to get file list (Status {response.status_code})")
            return []
    except Exception as e:
        print(f"  [API Error] Request failed: {e}")
        return []

def download_one_file():
    print(f"Start downloading samples from {REPO_ID} (using API compatible mode)...")
    
    all_keys = []
    for category, items in target_structure.items():
        all_keys.extend(items.keys())
    all_keys = list(set(all_keys))
    if 'csharp' in all_keys:
        all_keys.remove('csharp')
        all_keys.append('c%23')

    for folder_name in all_keys:
        folder_path = f"data/{folder_name}"
        
        print(f"\nProcessing: {folder_name} (Folder: {folder_name})")
        
        # --- Change point: Use HTTP API to get file list ---
        files = list_files_via_api(REPO_ID, folder_path)
        
        # Filter out .jsonl files
        target_files = [f for f in files if f.endswith(".jsonl")]
        
        if not target_files:
            print(f"  [Skip] No .jsonl file found in {folder_path}.")
            continue
            
        # Only take the first one
        file_to_download = target_files[0]
        print(f"  Found file: {file_to_download}")
        
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=file_to_download,
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False
            )
            print(f"  [Success] Downloaded to: {local_path}")
            # Move local_path to save_path
            # shutil.move(local_path, save_path)
        except Exception as e:
            print(f"  [Error] {e}")

if __name__ == "__main__":
    download_one_file()