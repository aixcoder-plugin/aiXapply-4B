# Experiments Overview

This directory contains the scripts used to reproduce the main experiment groups discussed in Section 7 of the paper. The layout is organized by editing paradigm and evaluation stage, so it is easy to compare full-file apply against alternative patch representations and then score all outputs with a shared evaluation pipeline.

## How This Maps to the Paper

- **RQ1: Why full-file apply?** Compare full-file apply against `unified diff` and `search-and-replace`.
- **RQ2: Feasibility and quality** Run the main apply benchmark and summarize per-language accuracy.
- **RQ3: Efficiency and cost** Measure latency / throughput for FastApply-style deployment.
- **RQ4: Training methods** Training scripts live under `training/`, while the prediction analysis and result aggregation live here.

For a consolidated write-up of the main numbers and conclusions, see:

- [Experiment Results](RESULTS.md)
- [实验结果汇总](RESULTS_CN.md)

## Directory Guide

| Path | Purpose | Key entrypoints |
| --- | --- | --- |
| `experiments/aiXapply/` | Main full-file apply inference scripts for `aiXapply` and comparable OpenAI-compatible models on the apply benchmark. Used for the overall and per-language accuracy tables. | `infer_openai.py` |
| `experiments/fast-apply/` | FastApply inference and serving utilities for low-latency / high-throughput experiments. Used for the speed table and deployment-oriented evaluation. | `infer_openai_fast-apply.py`, `vllm_start_fastapply7b.sh` |
| `experiments/apply_diff/` | Unified-diff baseline experiments. One script asks a model to generate unified diffs; another evaluates an apply model on diff-style inputs and converts outputs into the common evaluation format. | `infer_openai_generate_udiff.py`, `infer_openai_only_apply_udiff.py` |
| `experiments/search_and_replace/` | Search-and-replace baseline experiments, including prompt generation, local patch application, and error statistics for failed SAR edits. | `infer_openai_sar.py`, `prompt_opencode_sar.py`, `search_and_replace_tool.py`, `stat_sar_errors.py` |
| `experiments/evaluation/` | Shared evaluation pipeline: exact-match scoring, 6-class error taxonomy, optional LLM-assisted judging, and result summarization utilities. | `run_evaluation.py`, `judge_apply_correctness.py`, `classify_errors.py`, `summarize_apply_correctness_by_language.py` |

## Setup

Run commands from the repository root:

```bash
python -m pip install -r requirements.txt
```

Most scripts require explicit input data paths. Generate the final test parquet with `data_generation/verl_processor.py` or pass the location of an equivalent public benchmark file.

OpenAI-compatible inference scripts read credentials from CLI flags or environment variables:

```bash
export OPENAI_KEY=...              # shared baseline provider key
export VOLCANO_ENGINE_API_KEY=...  # VolcanoEngine preset
export OPENAI_BASE_URL=...         # optional custom endpoint
```

## Typical Workflows

### 1. Main full-file apply benchmark

```bash
python experiments/aiXapply/infer_openai.py \
  --data-path aiXapply_test_data/main_test_data.parquet \
  --provider local

python experiments/evaluation/run_evaluation.py \
  -i predictions/<model>_<timestamp>.jsonl
```

Optionally add `--classify_errors` to write error-type breakdowns under `experiments/evaluation/results/`.
To evaluate the other generalization sets on Hugging Face, replace `main_test_data.parquet` in `aiXapply_test_data/main_test_data.parquet` with the corresponding file from the Hugging Face dataset.

### 2. RQ1 editing-paradigm comparison

```bash
python experiments/apply_diff/infer_openai_generate_udiff.py \
  --data-path aiXapply_test_data/main_test_data.parquet \
  --provider custom \
  --api-key "$OPENAI_KEY" \
  --base-url "$OPENAI_BASE_URL" \
  --model <model-name>

python experiments/search_and_replace/infer_openai_sar.py \
  --data-path aiXapply_test_data/main_test_data.parquet \
  --provider custom \
  --api-key "$OPENAI_KEY" \
  --base-url "$OPENAI_BASE_URL" \
  --model <model-name>
```

Evaluate each output with `experiments/evaluation/run_evaluation.py`.

### 3. FastApply latency benchmark

```bash
WEIGHT_DIR=/path/to/FastApply-7B-v1.0 \
SERVE_MODEL_NAME=FastApply-7B-v1.0 \
PORT=8001 \
bash experiments/fast-apply/vllm_start_fastapply7b.sh

python experiments/fast-apply/infer_openai_fast-apply.py \
  --input aiXapply_test_data/main_test_data.parquet \
  --api-key EMPTY \
  --base-url http://127.0.0.1:8001/v1 \
  --model FastApply-7B-v1.0
```

Use scripts under `experiments/evaluation/` to check correctness and summarize results.

## Notes

- Most experiment scripts call an OpenAI-compatible endpoint and write prediction files in `.jsonl` format.
- Most dataset and endpoint defaults are now repo-relative or environment-driven, but you should still review CLI arguments and environment variables before reproducing large runs.
- For evaluation internals and the error taxonomy, see `experiments/evaluation/README.md`.
