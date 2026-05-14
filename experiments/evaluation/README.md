# Evaluation Utilities

This directory contains the shared evaluation tools used by the experiment scripts. The tools score full-file apply predictions, optionally classify wrong outputs into error types, and summarize LLM-judged correctness by language.

Run all commands from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Input Format

Most CLIs consume prediction `.jsonl` files. Each line should be a JSON object with these fields:

```json
{
  "index": 0,
  "prediction": "<update_file>...</update_file>",
  "elapsed_time": 1.23,
  "ground_truth": "...",
  "original_code": "...",
  "update_snippet": "...",
  "language": "python"
}
```

`run_evaluation.py` requires the fields above. Other tools accept common aliases such as `final_code`, `new_code`, `source_file`, or `output`, but using the normalized schema is recommended.

## Entrypoints

| Script | Purpose | Typical output |
| --- | --- | --- |
| `run_evaluation.py` | Score prediction files with the rule-based evaluator. Optionally classify wrong samples with `--classify_errors`. | `experiments/evaluation/results/<input_stem>/final_stats.json` |
| `classify_errors.py` | Standalone error-type classification for prediction JSONL files. | `predictions/classified/*.classified.jsonl` by default |
| `judge_apply_correctness.py` | Optional LLM-assisted correctness judging for JSONL files in a directory. | `*.judged.jsonl`, `*.summary.json`, `overall_summary.json` |
| `summarize_apply_correctness_by_language.py` | Convert judged files into a per-language accuracy table. | Markdown table on stdout or `--output` |

## Basic Scoring

Evaluate one or more prediction files:

```bash
python experiments/evaluation/run_evaluation.py \
  -i predictions/model_a.jsonl predictions/model_b.jsonl
```

For each input file, the script writes:

- `experiments/evaluation/results/<input_stem>/final_stats.json`
- Optional per-sample artifacts under the same directory when `--save_all` is enabled.

`--save_all` writes wrong-sample files and removes legacy files named `prediction.py`, `ground_truth.py`, `original_code.py`, `update_snippet.py`, and `diff.txt` inside the generated sample directories to avoid mixing stale artifacts.

## Error Classification

Add `--classify_errors` to classify wrong samples into a fixed six-class taxonomy:

```bash
python experiments/evaluation/run_evaluation.py \
  -i predictions/model.jsonl \
  --classify_errors
```

This writes `error_classification.jsonl` next to `final_stats.json`. Each row includes:

- final fields: `error_type`, `confidence`, `needs_review`, `reason_brief`
- rule-side fields: `rule_error_type`, `rule_confidence`, `rule_reason_brief`
- LLM fields when enabled: `used_llm`, `llm_error_type`, `llm_confidence`, `llm_reason_brief`

You can also run the standalone classifier:

```bash
python experiments/evaluation/classify_errors.py \
  -i predictions/model.jsonl \
  -o predictions/classified
```

By default it classifies only wrong samples. Use `--no-only_errors` to classify every row.

## Optional LLM Judging

LLM calls use an OpenAI-compatible Chat Completions endpoint. Configure the endpoint with environment variables or CLI arguments:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-4o-mini
```

Enable LLM-assisted error classification:

```bash
python experiments/evaluation/run_evaluation.py \
  -i predictions/model.jsonl \
  --classify_errors \
  --llm
```

Run correctness judging over a directory of prediction files:

```bash
python experiments/evaluation/judge_apply_correctness.py \
  --input-dir predictions \
  --input-pattern "*.jsonl" \
  --output-dir experiments/evaluation/results/apply_correctness_judge \
  --overwrite
```

Use `--dry-run` to validate input discovery and output structure without calling an LLM.

## Summarizing Judged Outputs

After `judge_apply_correctness.py`, generate a per-language table:

```bash
python experiments/evaluation/summarize_apply_correctness_by_language.py \
  --input-dir experiments/evaluation/results/apply_correctness_judge \
  --pattern "*.judged.jsonl"
```

Use `--output path/to/table.md` to save the table.

## Error Taxonomy

The classifier emits exactly one of these labels:

- `OUTPUT_INVALID`: output is not a usable full-file result, for example missing wrapper tags, extra narrative text, placeholder leakage, or invalid structured JSON/YAML/XML/INI.
- `PATCH_NOT_APPLIED`: expected changes are largely absent.
- `PATCH_INCOMPLETE`: some expected changes are present, but at least one required line or block is missing.
- `PATCH_INCORRECT`: the patch area was edited, but the logic or tokens do not match the expected change.
- `WRONG_POSITION`: correct-looking content appears in the wrong location or the target location was not updated correctly.
- `OUT_OF_PATCH_SIDE_EFFECT`: the expected patch is correct or mostly correct, but unrelated areas changed.

`OUTPUT_INVALID` is deterministic and handled by rules in `update_file_parser.py`. The other labels use rule features and can optionally be refined by the LLM path.

## Notes

- The evaluator depends on `training.rl.reward_function_rule_based_multi.compute_score_pygments`, so run it from the repository root.
- LLM-based classification and judging can be slow and cost money. Start with rule-only runs, then enable `--llm` for ambiguous samples or final analysis.
- All result directories are generated artifacts and should be reviewed before committing.
