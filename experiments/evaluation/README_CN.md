# 评测工具说明

本目录包含实验脚本共用的评测工具，用于给 full-file apply 预测结果打分、对错误样本进行错误类型分类，并按语言汇总可选的 LLM 判定结果。

请在仓库根目录运行命令：

```bash
python -m pip install -r requirements.txt
```

## 输入格式

大多数 CLI 读取预测结果 `.jsonl` 文件。每一行应是一个 JSON 对象，推荐字段如下：

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

`run_evaluation.py` 需要上述字段。其他工具兼容一些常见别名，例如 `final_code`、`new_code`、`source_file`、`output`，但开源复现时建议统一使用推荐字段。

## 脚本入口

| 脚本 | 用途 | 主要输出 |
| --- | --- | --- |
| `run_evaluation.py` | 使用规则评测器给预测文件打分；可通过 `--classify_errors` 对错误样本分类。 | `experiments/evaluation/results/<input_stem>/final_stats.json` |
| `classify_errors.py` | 独立的错误类型分类 CLI。 | 默认写入 `predictions/classified/*.classified.jsonl` |
| `judge_apply_correctness.py` | 使用 OpenAI-compatible LLM 对目录内 JSONL 做 correctness judging。 | `*.judged.jsonl`、`*.summary.json`、`overall_summary.json` |
| `summarize_apply_correctness_by_language.py` | 将 judged 文件汇总成按语言分组的准确率表。 | stdout 中的 Markdown 表，或通过 `--output` 保存 |

## 基础评测

评测一个或多个预测文件：

```bash
python experiments/evaluation/run_evaluation.py \
  -i predictions/model_a.jsonl predictions/model_b.jsonl
```

每个输入文件会生成：

- `experiments/evaluation/results/<input_stem>/final_stats.json`
- 如果启用 `--save_all`，还会在同目录下写入错误样本的预测、标准答案、原始代码、更新片段和 diff。

注意：`--save_all` 会清理生成样本目录中的旧文件名 `prediction.py`、`ground_truth.py`、`original_code.py`、`update_snippet.py`、`diff.txt`，避免新旧产物混在一起。

## 错误类型分类

在基础评测上加入 `--classify_errors`，可将错误样本分到固定的 6 类错误类型中：

```bash
python experiments/evaluation/run_evaluation.py \
  -i predictions/model.jsonl \
  --classify_errors
```

输出文件 `error_classification.jsonl` 会和 `final_stats.json` 放在同一结果目录。每行包含：

- 最终分类字段：`error_type`、`confidence`、`needs_review`、`reason_brief`
- 规则侧字段：`rule_error_type`、`rule_confidence`、`rule_reason_brief`
- 启用 LLM 时的字段：`used_llm`、`llm_error_type`、`llm_confidence`、`llm_reason_brief`

也可以单独运行分类器：

```bash
python experiments/evaluation/classify_errors.py \
  -i predictions/model.jsonl \
  -o predictions/classified
```

默认只分类错误样本。使用 `--no-only_errors` 可以对所有样本分类。

## 可选 LLM 判定

LLM 调用使用 OpenAI-compatible Chat Completions 接口。可以通过环境变量或 CLI 参数配置：

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-4o-mini
```

在错误分类中启用 LLM：

```bash
python experiments/evaluation/run_evaluation.py \
  -i predictions/model.jsonl \
  --classify_errors \
  --llm
```

对目录中的预测文件进行 correctness judging：

```bash
python experiments/evaluation/judge_apply_correctness.py \
  --input-dir predictions \
  --input-pattern "*.jsonl" \
  --output-dir experiments/evaluation/results/apply_correctness_judge \
  --overwrite
```

如果只想检查输入发现和输出结构，可以加 `--dry-run`，不会调用 LLM。

## 汇总 judged 结果

运行 `judge_apply_correctness.py` 后，可以按语言生成表格：

```bash
python experiments/evaluation/summarize_apply_correctness_by_language.py \
  --input-dir experiments/evaluation/results/apply_correctness_judge \
  --pattern "*.judged.jsonl"
```

使用 `--output path/to/table.md` 可以保存表格。

## 错误类型

分类器只会输出以下标签之一：

- `OUTPUT_INVALID`：输出不能作为可用的完整文件，例如缺少 wrapper、包含额外解释、占位符泄露，或 JSON/YAML/XML/INI 等结构化格式无法解析。
- `PATCH_NOT_APPLIED`：期望修改基本没有被应用。
- `PATCH_INCOMPLETE`：应用了部分期望修改，但至少缺少一行或一个代码块。
- `PATCH_INCORRECT`：修改发生在补丁区域，但逻辑或 token 与期望修改不一致。
- `WRONG_POSITION`：看起来正确的内容出现在错误位置，或目标位置没有被正确更新。
- `OUT_OF_PATCH_SIDE_EFFECT`：期望修改正确或基本正确，但修改了补丁区域之外的无关内容。

`OUTPUT_INVALID` 是确定性格式错误，由 `update_file_parser.py` 中的规则处理。其他类型基于 diff 特征和规则候选分类，也可以通过 LLM 路径进一步判断。

## 注意事项

- 评测器依赖 `training.rl.reward_function_rule_based_multi.compute_score_pygments`，请从仓库根目录运行。
- LLM 分类和判定会增加耗时和成本。建议先运行纯规则评测，再只对需要分析的样本启用 `--llm`。
- `experiments/evaluation/results/` 下的内容是生成产物，提交前请人工确认是否需要纳入版本控制。
