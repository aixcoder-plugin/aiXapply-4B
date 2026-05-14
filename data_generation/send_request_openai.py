"""
aiXapply data generation pipeline.

Stage: LLM request dispatch.
Purpose: send description and verification batches to an OpenAI-compatible API.
Inputs: batch_*.jsonl files grouped by language.
Outputs: output_batch_*.jsonl files grouped by language.
"""
import os
import glob
import json
import asyncio
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from aiolimiter import AsyncLimiter
import argparse
import logging

from config import BATCH_REQUEST_CONFIG
from openai import AsyncOpenAI
from aiohttp import ClientError, ClientResponseError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# Configure the OpenAI-compatible client via environment variables.
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
)

# It limits the rate of asynchronous operations to 120 requests per 60 seconds
rate_limiter = AsyncLimiter(120, 60)


async def generate_update_openai(messages, config_params, max_tokens, custom_id):
    """Generate using OpenAI-compatible API (DeepSeek/SiliconFlow)."""
    max_retries = 2
    attempt = 0
    backoff_factor = 2

    while attempt <= max_retries:
        try:
            async with rate_limiter:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=config_params['model'],
                        messages=messages,
                        stream=False,
                        temperature=config_params['temperature'],
                        top_p=config_params['top_p'],
                        max_tokens=max_tokens
                    ),
                    timeout=2000
                )
                return response.choices[0].message.content
        except asyncio.TimeoutError:
            logging.error(f"Timeout after 2000s for messages: {custom_id}...")
            return "DELETE_ROW"
        except ClientResponseError as cre:
            status = cre.status
            logging.error(f"HTTP error {status} for messages: {custom_id}... - {cre.message}")
            if status in [400, 401, 402, 422]:
                return "DELETE_ROW"
            elif status == 429:
                logging.warning("429 - Rate Limit Reached: Pacing requests.")
            elif status in [500, 503]:
                logging.warning(f"{status} - Server Error: Retrying after backoff.")
            else:
                return "DELETE_ROW"
        except ClientError as ce:
            logging.error(f"Client error: {ce} for messages: {custom_id}...")
        except Exception as e:
            logging.error(f"Unexpected error: {e} for messages: {custom_id}...")

        attempt += 1
        if attempt <= max_retries:
            wait_time = backoff_factor ** attempt + 10
            logging.info(f"Retrying in {wait_time} seconds... (Attempt {attempt} of {max_retries})")
            await asyncio.sleep(wait_time)
    
    logging.error(f"All retries failed for messages: {custom_id}...")
    return "DELETE_ROW"


async def process_row(line, mode):
    """Process a single row of the DataFrame."""
    data = json.loads(line)
    
    user_prompt = data['user_prompt']
    custom_id = data['custom_id']
    max_tokens = data.get("max_tokens", 32768)
    
    # Determine config based on mode
    if mode == "description":
        config_params = BATCH_REQUEST_CONFIG['description_generation']
    elif mode == "code":
        config_params = BATCH_REQUEST_CONFIG['code_generation']
    else:
        config_params = BATCH_REQUEST_CONFIG['verification']
        max_tokens = 8192

    messages = [
        {"role": "system", "content": config_params['system_prompt']},
        {"role": "user", "content": user_prompt}
    ]
    
    # Use different API based on mode
    response = await generate_update_openai(messages, config_params, max_tokens, custom_id)

    if response == "DELETE_ROW":
        return None

    result = {
        "custom_id": custom_id,
        "response": {
            "body": {
                "choices": [{"index": 0, "message": {"content": response}}]
            }
        }
    }
    
    return result


async def main(batch_dir, output_dir, mode):
    """Main processing function."""
    # Find all batch files
    input_files = sorted(glob.glob(os.path.join(batch_dir, "**/batch_*.jsonl"), recursive=True))

    input_files_output = glob.glob(os.path.join(output_dir, "**/output_batch_*.jsonl"), recursive=True)

    input_files_output = [
                            input_file.replace("/output_batch_", "/batch_").replace(output_dir,batch_dir)
                            for input_file in input_files_output
                    ]

    print("Origin input files:", len(input_files))
    input_files = [file for file in input_files if file not in input_files_output]
    print("Already output files:", len(input_files_output))
    print("Finally input files:",len(input_files))

    # Group files by language
    language_type_dict = {}
    for input_file in tqdm(input_files, desc="Processing input files", unit="file"):
        # Extract language from path: .../batch_data/<language>/batch_xxx.jsonl
        path_parts = input_file.split(os.sep)
        batch_data_idx = -1
        for i, part in enumerate(path_parts):
            if part == "batch_data":
                batch_data_idx = i
                break
        
        if batch_data_idx != -1 and batch_data_idx + 1 < len(path_parts):
            language_type = path_parts[batch_data_idx + 1]
        else:
            # Fallback: use directory name before the file
            language_type = os.path.basename(os.path.dirname(input_file))
        
        if language_type not in language_type_dict:
            language_type_dict[language_type] = []
        language_type_dict[language_type].append(input_file)

    logging.info(f"Found languages: {list(language_type_dict.keys())}")
    logging.info(f"Total files: {len(input_files)}")

    print("--------------------------------")
    print(language_type_dict.keys())
    print("--------------------------------")

    for language_type, files in tqdm(
                            language_type_dict.items(),
                            desc=f"Processing {language_type}",
                            unit="language"
                        ):
        output_path = os.path.join(output_dir, language_type)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        logging.info(f"Processing {language_type}: {len(files)} files")

        for input_file in tqdm(
            sorted(files),
            desc=f"Processing {language_type}",
            unit="file",
            leave=False
        ):
            batch_number = os.path.basename(input_file).replace("batch_", "").replace(".jsonl", "")
            output_file = os.path.join(output_path, f"output_batch_{batch_number}.jsonl")

            if os.path.exists(output_file):
                logging.debug(f"Skipping existing file: {output_file}")
                continue

            # Read input file
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Process all rows. gather() keeps result order aligned with input lines.
            tasks = [asyncio.create_task(process_row(line, mode)) for line in lines]
            results = [result for result in await asyncio.gather(*tasks) if result is not None]

            # Write results
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            logging.info(f"Saved {len(results)} results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-language batch request processor")
    
    # Description mode paths
    parser.add_argument(
        "--description_batch_dir",
        default="dataset/synthetic_data/batch_data",
        type=str,
        help="Directory containing batch input files for description generation"
    )
    parser.add_argument(
        "--description_output_dir",
        default="dataset/synthetic_data/description",
        type=str,
        help="Directory to save description output files"
    )
    
    # Verification mode paths
    parser.add_argument(
        "--judge_batch_dir",
        default="dataset/synthetic_data/codes_processed",
        type=str,
        help="Directory containing batch input files for verification"
    )
    parser.add_argument(
        "--judge_output_dir",
        default="dataset/synthetic_data/judge_result",
        type=str,
        help="Directory to save verification output files"
    )
    
    parser.add_argument(
        "-m", "--mode",
        default="description",
        choices=["description", "verification"],
        help="Mode to run the script"
    )
    
    args = parser.parse_args()
 
    # Set paths based on mode
    if args.mode == "description":
        args.batch_dir = args.description_batch_dir
        args.output_dir = args.description_output_dir
    elif args.mode == "verification":
        args.batch_dir = args.judge_batch_dir
        args.output_dir = args.judge_output_dir

    # Validate paths
    if not os.path.exists(args.batch_dir):
        logging.error(f"Error: Batch directory {args.batch_dir} does not exist.")
        exit(1)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.info(f"Mode: {args.mode}")
    logging.info(f"Batch directory: {args.batch_dir}")
    logging.info(f"Output directory: {args.output_dir}")

    asyncio.run(main(
        args.batch_dir,
        args.output_dir,
        args.mode
    ))
