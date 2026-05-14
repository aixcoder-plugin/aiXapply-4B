# Multi-Language Data Generation Pipeline

This directory contains the complete pipeline for generating the aiXapply multi-language training dataset.

## Directory Structure

```
data_generation/
├── download_dataset.py               # Optional pre-step: download CommitPack source data
├── prepare_batch_data.py             # Step 1: sample data and prepare description batches
├── send_request_openai.py            # Step 2/6: send description and LLM-as-judge requests
├── generate_code.py                  # Step 3: prepare code-generation batches
├── send_request_claude_stream.py     # Step 4: send Claude code-generation requests
├── batch_processor.py                # Step 5: prepare verification batches
├── judge_processor.py                # Step 7: process verification results
├── infer_openai.py                   # Step 8: run multi-model inference verification
├── extract_jsonl_data.py             # Step 9: extract evaluation results
├── llm_filter.py                     # Step 10: filter low-quality samples
├── verl_processor.py                 # Step 11: convert to veRL training format
├── config.py                         # Configuration file
├── prompt.py                         # Prompt templates
├── sample_config.json                # Sampling configuration for each language
└── sample_config_split.json          # Train/test split configuration
```

## Supported Languages

| Category | Language/Format | Description |
|----------|-----------------|-------------|
| **General Programming Languages** | Python, Java, C++, C, Go, Rust, TypeScript, JavaScript | For building application logic |
| **Script/Shell** | Shell | For automation and system administration |
| **Data Serialization/Configuration** | JSON, YAML, INI, XML | For data exchange or configuration parameters |
| **Markup/Documentation Languages** | Markdown, reStructuredText, HTML | For typesetting and documentation |
| **Domain-Specific Languages (DSL)** | SQL, Dockerfile, Makefile | Specialized for specific tools or domains |

## Data Generation Pipeline

The entire pipeline is divided into the following phases:

### Phase 1: Data Preparation

Optional source dataset download:

```bash
python data_generation/download_dataset.py
```

Pipeline steps:

```bash
# Step 1: Data sampling and batch preparation
python data_generation/prepare_batch_data.py \
    --config data_generation/sample_config.json \
    --output_dir dataset/synthetic_data/batch_data
```

### Phase 2: Description Generation

Use an OpenAI-compatible LLM endpoint to generate detailed modification descriptions for code changes.

```bash
# Step 2: Send description generation requests
python data_generation/send_request_openai.py \
    --mode description
```

### Phase 3: Code Generation

Use LLM (e.g., Claude) to generate `update_snippet` and `final_code`.

```bash
# Step 3: Multi-language code generation data preparation
python data_generation/generate_code.py \
    --describe_dir dataset/synthetic_data/description \
    --output_dir dataset/synthetic_data/description_combined

# Step 4: Send requests
python data_generation/send_request_claude_stream.py \
    --mode code
```

### Phase 4: Verification and Filtering

Verify whether the generated code correctly applies the update snippets.

```bash
# Step 5: Prepare verification requests
python data_generation/batch_processor.py 

# Step 6: LLM as Judge to evaluate generated dataset quality
python data_generation/send_request_openai.py \
    --mode verification

# Step 7: Process verification results
python data_generation/judge_processor.py \
    --merge

# Step 8: Multi-model inference verification
# Supported providers: VolcanoEngine and bailian.
python data_generation/infer_openai.py \
    --provider VolcanoEngine \
    --data-dir dataset/synthetic_data/judge_processed \
    --output-dir dataset/synthetic_data/predictions

python data_generation/infer_openai.py \
    --provider bailian \
    --data-dir dataset/synthetic_data/judge_processed \
    --output-dir dataset/synthetic_data/predictions

# Step 9: Extract evaluation results and calculate accuracy
python data_generation/extract_jsonl_data.py \
    --predictions-dir dataset/synthetic_data/predictions \
    --method pygments \
    --save-result dataset/synthetic_data/predictions_result/result_dict.json

# Step 10: Remove erroneous samples based on evaluation results
python data_generation/llm_filter.py \
    --result-file dataset/synthetic_data/predictions_result/result_dict.json \
    --data-dir dataset/synthetic_data/judge_processed \
    --output-dir dataset/synthetic_data/filtered \
    --method pygments \
    --dynamic-threshold
```


### Phase 5: Dataset Construction

```bash
# Step 11: Generate veRL training format
python data_generation/verl_processor.py \
    --output_dir dataset/synthetic_data/final_dataset
```

## Configuration Files

### sample_config.json

Defines the sampling count for each language:

```json
{
    "GENERAL_PROGRAMMING": {
        "python": 9341,
        "javascript": 8182,
        ...
    },
    "SCRIPT_SHELL": { ... },
    "DATA_CONFIG": { ... },
    "MARKUP_DOC": { ... },
    "DSL": { ... }
}
```

### sample_config_split.json

Defines the train/test split:
Note: The test data count is determined by the values specified in the JSON file. After splitting the test set, all remaining data is assigned to the training set.

```json
{
    "GENERAL_PROGRAMMING": {
        "python": { "train": 8297, "test": 195 },
        "javascript": { "train": 7238, "test": 200 },
        ...
    },
    "SCRIPT_SHELL": { ... },
    "DATA_CONFIG": { ... },
    "MARKUP_DOC": { ... },
    "DSL": { ... }
}
```

## Prompt Templates

Data generation uses a three-stage prompt approach:

1. **PROMPT_DESCRIPTION**: Analyze code differences and generate modification descriptions
2. **PROMPT_CODE_GENERATION**: Generate `update_snippet` and `final_code` based on descriptions
3. **PROMPT_VERIFICATION**: Verify the correctness of generated results

See `prompt.py` for details.

## Data Filtering Criteria

Filtering rules defined in `prepare_batch_data.py`:

| Parameter | Range/Condition |
|-----------|-----------------|
| Character count | 45 - 70,000 |
| Line count | 20 - 1,400 |
| New/old code line ratio | 0.7 - 1.3 |
| Commit message length | ≥ 10 |
| Minimum code lines | > 50 |

## API Configuration

Configure API parameters in `config.py`. The model names can be overridden with environment variables:

```python
BATCH_REQUEST_CONFIG = {
    "description_generation": {
        "model": os.getenv("DESCRIPTION_MODEL", "deepseek-v3-2-251201"),
        "temperature": 0.3,
        ...
    },
    "code_generation": {
        "model": os.getenv("CODE_GENERATION_MODEL", "claude-opus-4-5-20251101"),
        "temperature": 0.3,
        ...
    },
    "verification": {
        "model": os.getenv("VERIFICATION_MODEL", "deepseek-v3-2-251201"),
        ...
    }
}
```

Required credentials:

```bash
# Step 2 and Step 6, OpenAI-compatible endpoint
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...

# Step 4, Claude code generation
export ANTHROPIC_API_KEY=...

# Step 8, provider inference
export VOLCANO_ENGINE_API_KEY=...
export BAILIAN_API_KEY=...
```

## Output Format

Final generated dataset format (veRL training format):

```json
{
    "data_source": "verl/aiXapply",
    "prompt": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ],
    "reward_model": {
        "style": "rule",
        "ground_truth": "..."
    },
    "extra_info": {
        "language": "",
        "index": 0,
        "update_snippet": "...",
        ...
    }
}
```
