"""
aiXapply data generation pipeline.

Stage: Claude request dispatch.
Purpose: send code-generation batches to the Anthropic Messages API.
Inputs: batch_*.jsonl files grouped by language.
Outputs: output_batch_*.jsonl files grouped by language.
"""
import os
import glob
import json
import asyncio
import argparse
import logging
from typing import Dict, List, Any

import anthropic
from tqdm.asyncio import tqdm

from config import BATCH_REQUEST_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

REPLACE_DICT = [
    ("c++", 'cpp')
]


async def send_single_request(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    custom_id: str,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    pbar: tqdm = None
) -> Dict[str, Any]:
    """Send a single streaming request to Claude API."""
    async with semaphore:
        try:
            # Use streaming to get response
            response_text = ""
            async with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            ) as stream:
                async for text in stream.text_stream:
                    response_text += text
            
            if pbar:
                pbar.update(1)
            
            return {
                "custom_id": custom_id,
                "response": response_text,
                "success": True
            }
        except Exception as e:
            logging.error(f"Failed to process request {custom_id}: {e}")
            if pbar:
                pbar.update(1)
            return {
                "custom_id": custom_id,
                "response": None,
                "success": False,
                "error": str(e)
            }


async def process_file_streaming(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    lines: List[str],
    output_file: str,
    subdir: str = ""
):
    """Process a file using streaming API with concurrent requests."""
    tasks = []
    
    for line in lines:
        data = json.loads(line)
        
        # Simplified: {custom_id, user_prompt, max_tokens}
        user_prompt = data['user_prompt']
        custom_id = data['custom_id'].replace(REPLACE_DICT[0][0], REPLACE_DICT[0][1])
        max_tokens = min(data.get("max_tokens", 2048), 64000)  # Claude API max limit is 64000
        
        # Determine type
        if '<changed_code>' in user_prompt:
            # Description generation params
            config_params = BATCH_REQUEST_CONFIG['description_generation']
        elif '<brief_change>' in user_prompt:
            # Code generation params
            config_params = BATCH_REQUEST_CONFIG['code_generation']
        else:
            # Verification params
            config_params = BATCH_REQUEST_CONFIG['verification']
        
        task = send_single_request(
            client=client,
            semaphore=semaphore,
            custom_id=custom_id,
            system_prompt=config_params['system_prompt'],
            user_prompt=user_prompt,
            model=config_params['model'],
            temperature=config_params['temperature'],
            max_tokens=max_tokens
        )
        tasks.append(task)
    
    # Run all tasks concurrently with semaphore limiting concurrency
    results = await asyncio.gather(*tasks)
    
    # Write results to output file
    successful_results = [r for r in results if r["success"]]
    failed_count = len(results) - len(successful_results)
    
    if failed_count > 0:
        logging.warning(f"Failed {failed_count} requests for {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in successful_results:
            f.write(json.dumps({
                "custom_id": result["custom_id"].replace(REPLACE_DICT[0][1], REPLACE_DICT[0][0]),
                "response": {
                    "body": {
                        "choices": [{
                            "index": 0,
                            "message": {
                                "content": result["response"]
                            }
                        }]
                    }
                }
            }, ensure_ascii=False) + '\n')
    
    logging.info(f"Completed {output_file}: {len(successful_results)}/{len(results)} successful")
    return len(successful_results), len(results)


async def main(client: anthropic.AsyncAnthropic, batch_dir: str, output_dir: str, mode: str = None, subdir: str = "", max_concurrency: int = 10):
    """Main function to process all files."""
    input_files = sorted(glob.glob(os.path.join(batch_dir, "**/batch_*.jsonl"), recursive=True))
    input_files_output = glob.glob(os.path.join(output_dir, "**/output_batch_*.jsonl"), recursive=True)

    input_files_output = [
        input_file.replace("/output_batch_", "/batch_").replace(output_dir, batch_dir)
        for input_file in input_files_output
    ]
    print("--------------------------------")
    logging.info(f"Origin input files: {len(input_files)}")
    input_files = [file for file in input_files if file not in input_files_output]
    logging.info(f"Already output files: {len(input_files_output)}")
    logging.info(f"Finally input files: {len(input_files)}")
    print("--------------------------------")

    # Group files by language type
    language_type_dict = {}
    for input_file in input_files:
        language_type = input_file.split("/")[-2]
        if language_type not in language_type_dict:
            language_type_dict[language_type] = []
        language_type_dict[language_type].append(input_file)

    for language_type, count in language_type_dict.items():
        logging.info(f"{language_type}: {len(count)} files")

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrency)
    
    total_successful = 0
    total_requests = 0

    for language_type, files in tqdm(language_type_dict.items(), 
                                      desc="Processing language types",
                                      unit="language type",
                                      total=len(language_type_dict)):
        output_path = os.path.join(output_dir, language_type)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        logging.info(f"Processing {language_type}")
        for input_file in tqdm(files,
                               desc=f"Processing input files for {language_type}",
                               unit="file",
                               total=len(files)):
            # Generate output file name
            batch_number = input_file.split("/")[-1].split("_")[-1].split(".")[0]
            output_file = os.path.join(output_path, f"output_batch_{batch_number}.jsonl")

            # Read input file
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if not lines:
                logging.warning(f"No data in {input_file}, skipping.")
                continue

            # Process file with streaming
            successful, total = await process_file_streaming(
                client=client,
                semaphore=semaphore,
                lines=lines,
                output_file=output_file,
                subdir=subdir
            )
            total_successful += successful
            total_requests += total
    
    logging.info(f"Total: {total_successful}/{total_requests} requests successful")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send streaming requests to Claude API.")
    
    # Subdir selection (e.g., csharp, long, slice)
    parser.add_argument(
        "--subdirs",
        nargs="+",
        default=None,
        type=str,
        help="Subdirectory names to process (e.g., csharp long slice). If not specified, processes all subdirs."
    )
    
    # Base data directory
    parser.add_argument(
        "--base_dir",
        default="dataset/synthetic_data",
        type=str,
        help="Base directory for extra synthetic data"
    )
    
    parser.add_argument("-m", "--mode", 
                        default="code", 
                        choices=["description", "code", "verification"], 
                        help="Mode to run the script")
    parser.add_argument("--api_key", 
                        type=str, 
                        default=os.getenv("ANTHROPIC_API_KEY", ""),
                        help="Anthropic credential (or set ANTHROPIC_API_KEY in the environment)")
    parser.add_argument("--timeout", 
                        type=float, 
                        default=600.0,
                        help="Timeout in seconds for API requests (default: 600s)")
    parser.add_argument("--max_concurrency",
                        type=int,
                        default=10,
                        help="Maximum number of concurrent requests (default: 10)")

    args = parser.parse_args()

    # Define subdirectory mappings for each mode
    MODE_DIR_MAPPING = {
        "description": {
            "batch_subdir": "commitpack_combine",
            "output_subdir": "description"
        },
        "code": {
            "batch_subdir": "description_combined",
            "output_subdir": "codes"
        },
        "verification": {
            "batch_subdir": "codes_processed",
            "output_subdir": "judge_result"
        }
    }
    
    mode_config = MODE_DIR_MAPPING[args.mode]
    
    # Discover available subdirs if not specified
    if args.subdirs is None:
        # Auto-discover subdirs from base_dir
        potential_subdirs = []
        if os.path.exists(args.base_dir):
            for item in os.listdir(args.base_dir):
                item_path = os.path.join(args.base_dir, item)
                if os.path.isdir(item_path):
                    # Check if this subdir has the required batch_subdir
                    batch_path = os.path.join(item_path, mode_config["batch_subdir"])
                    if os.path.exists(batch_path):
                        potential_subdirs.append(item)
        args.subdirs = potential_subdirs if potential_subdirs else [""]
        logging.info(f"Auto-discovered subdirs: {args.subdirs}")

    # Initialize the Anthropic client with an optional base URL override.
    client_kwargs: Dict[str, Any] = {
        "api_key": args.api_key,
        "timeout": args.timeout,
    }
    anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if anthropic_base_url:
        client_kwargs["base_url"] = anthropic_base_url
    client = anthropic.AsyncAnthropic(**client_kwargs)

    # Ensure credentials are available before sending requests.
    if not client.api_key:
        logging.error(
            "Error: Anthropic credential not set. Please set ANTHROPIC_API_KEY or pass --api_key."
        )
        exit(1)

    logging.info(f"Mode: {args.mode}")
    logging.info(f"Base directory: {args.base_dir}")
    logging.info(f"Processing subdirs: {args.subdirs}")
    logging.info(f"Max concurrency: {args.max_concurrency}")
    
    # Process each subdir
    for subdir in tqdm(args.subdirs, desc="Processing subdirs", unit="subdir"):
        if subdir:
            batch_dir = os.path.join(args.base_dir, subdir, mode_config["batch_subdir"])
            output_dir = os.path.join(args.base_dir, subdir, mode_config["output_subdir"])
        else:
            # Fallback to old behavior if empty subdir
            batch_dir = os.path.join(args.base_dir, mode_config["batch_subdir"])
            output_dir = os.path.join(args.base_dir, mode_config["output_subdir"])
        
        logging.info(f"Processing subdir: {subdir if subdir else '(root)'}")
        logging.info(f"  Batch directory: {batch_dir}")
        logging.info(f"  Output directory: {output_dir}")
        
        # Validate paths
        if not os.path.exists(batch_dir):
            logging.warning(f"Batch directory {batch_dir} does not exist, skipping...")
            continue
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        asyncio.run(main(client, batch_dir, output_dir, args.mode, subdir, args.max_concurrency))
