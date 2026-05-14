# aiXapply 实验结果

本文档汇总 **aiXapply** 的主要实验结果，作为 `experiments/` 中可运行脚本的结果说明文档。内容覆盖编辑表示对比、主测试集准确率、泛化评测、错误分析、推理效率以及 SFT / RL 训练方式对比。

## 实验问题

实验围绕四个问题展开：

- **RQ1：为什么选择全文 Apply？** 比较 `unified diff`、`search-and-replace` 与全文 Apply。
- **RQ2：4B Apply 模型能否达到强基线准确率？** 在 20 种语言/文件格式上评测 aiXapply。
- **RQ3：全文 Apply 能否满足 IDE 场景的效率要求？** 测量 speculative decoding 下的延迟与吞吐。
- **RQ4：什么训练方式更有效？** 对比 SFT 与 RL / GRPO 在主分布和分布外场景中的表现。

## 实验设置

所有主实验统一采用结构化输入输出：

```text
<language>{language}</language>
<source_file>{source_file}</source_file>
<update_snippet>{update_snippet}</update_snippet>

-> <update_file>{full updated file}</update_file>
```

### 数据集

| 数据集 | 样本数 | 用途 |
| --- | ---: | --- |
| aiXapply 主测试集 | 1,637 | 覆盖 20 种语言/文件格式的主 benchmark。 |
| Random Placeholders | 1,637 | 测试对不同占位符格式的鲁棒性。 |
| Chunk File | 1,637 | 测试仅给定局部/截断源文件上下文时的 Apply 能力，模拟 IDE 选区修改。 |
| Long Context / Large File | 51 | 测试长上下文 / 大文件稳定性。 |
| Untrained Languages | 647 | 测试 PHP、CSS、SystemVerilog、C# 等训练外语言迁移。 |
| Fast-Apply 测试集 | 200 | 外部迁移测试；包含部分 noisy / ambiguous 样本。 |
| Diff-XYZ | 1,000 | 测试对 unified diff 输入格式的跨格式泛化能力。 |

### 主测试集语言分布

| 语言/格式 | 数量 | 占比 |
| --- | ---: | ---: |
| Java | 200 | 12.22% |
| JavaScript | 200 | 12.22% |
| Python | 195 | 11.91% |
| C | 130 | 7.94% |
| C++ | 128 | 7.82% |
| Go | 80 | 4.89% |
| JSON | 54 | 3.30% |
| XML | 50 | 3.05% |
| Shell | 50 | 3.05% |
| Markdown | 50 | 3.05% |
| Makefile | 50 | 3.05% |
| Text | 50 | 3.05% |
| INI | 50 | 3.05% |
| reStructuredText | 50 | 3.05% |
| Dockerfile | 50 | 3.05% |
| TypeScript | 50 | 3.05% |
| SQL | 50 | 3.05% |
| Rust | 50 | 3.05% |
| YAML | 50 | 3.05% |
| HTML | 50 | 3.05% |
| **总计** | **1,637** | **100.00%** |

### 评测指标

主指标为 **等价准确率**：

- 代码文件主要使用 Pygments token 级等价判断。
- JSON、YAML、XML、INI 等结构化格式会进行解析或归一化。
- 错误进一步划分为 `OUTPUT_INVALID`、`PATCH_NOT_APPLIED`、`PATCH_INCOMPLETE`、`PATCH_INCORRECT`、`WRONG_POSITION`、`OUT_OF_PATCH_SIDE_EFFECT`。

推理效率实验额外汇报：

- **平均延迟**：单请求平均耗时。
- **P95 延迟**：单请求耗时的 95 分位数，用于观察尾部时延。
- **平均输出 tokens**：每条样本平均生成 token 数。
- **吞吐**：总生成 token 数除以总耗时。
- **平均加速比**：相对于无 speculative decoding baseline 的平均时延加速比。

参数搜索阶段使用按语言分层抽样得到的 10% 子集（`165` 条，随机种子 `42`）。最终稳定性验证使用完整的 `1,637` 条主测试集。

## RQ1：编辑表示对比

代码片段融合至少有三种路线：

- `unified diff`：输出紧凑，但对 patch 语法和上下文严格敏感。
- `search-and-replace`：更短，适合 agent 工具循环，但依赖精确抄写原文边界。
- 全文 Apply：一次性准确率最高，但输出更长。

![代码 Apply 的准确率-延迟前沿](../assets/figures/aiXapply-latency-accuracy-frontier.png)

*图 1：unified diff、search-and-replace 与 full-file Apply 的准确率-延迟对比。aiXapply-RL 在保持全文 Apply 准确率优势的同时，将延迟降低到交互式可用范围。*

### 平均准确率

| 模型 | Unified diff | Search-and-replace | 全文 Apply |
| --- | ---: | ---: | ---: |
| DeepSeek-V3.2 | 0.560 | 0.749 | 0.916 |
| GLM-5 | 0.649 | 0.905 | 0.921 |
| Kimi-K2.5 | 0.734 | 0.821 | 0.971 |
| Qwen3.5-397B-A17B | 0.644 | 0.866 | 0.948 |

在 DeepSeek-V3.2 上，从 `search-and-replace` 切换到全文 Apply，准确率提升约 **17.5 个百分点**。这说明如果目标是最高的一次性融合准确率，全文 Apply 是更合适的目标表示。

### 输出长度与延迟

| 模型 | Diff tokens | Diff 延迟 | S&R tokens | S&R 延迟 | Apply tokens | Apply 延迟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-V3.2 | 462 | 14.22s | 926 | 28.48s | 3684 | 108.96s |
| GLM-5 | 437 | 12.98s | 512 | 13.38s | 3646 | 51.33s |
| Kimi-K2.5 | 483 | 19.04s | 596 | 22.64s | 3680 | 79.69s |
| Qwen3.5-397B | 460 | 5.82s | 902 | 11.21s | 3674 | 35.90s |
| aiXapply-RL | -- | -- | -- | -- | 3734 | 1.44s |

全文 Apply 的主要代价是生成内容更长。aiXapply 通过专用 4B 模型和任务匹配的 speculative decoding，将全文件生成的延迟降低到交互式范围。

## RQ2：主测试集准确率

下表对比 4B 基座模型、强基线、主流大模型、已有 Apply 模型以及 aiXapply 变体。

| 语言 | Qwen3-4B | DeepSeek-V3.2 | Qwen3.5-397B | Kimi-K2.5 | GLM-5 | aiXapply-RL | aiXapply-SFT | Fast-Apply-7B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C | 0.66 | 0.915 | 0.946 | 0.992 | 0.931 | 0.96 | 0.93 | 0.546 |
| C++ | 0.66 | 0.914 | 0.977 | 0.953 | 0.945 | 0.92 | 0.945 | 0.539 |
| Dockerfile | 0.68 | 0.94 | 1.00 | 1.00 | 0.96 | 1.00 | 1.00 | 0.74 |
| Go | 0.65 | 0.937 | 0.95 | 0.975 | 0.963 | 0.93 | 0.912 | 0.65 |
| HTML | 0.64 | 0.84 | 0.96 | 0.96 | 0.92 | 0.86 | 0.96 | 0.60 |
| INI | 0.50 | 0.88 | 0.94 | 0.96 | 0.98 | 0.92 | 0.96 | 0.64 |
| Java | 0.71 | 0.955 | 0.945 | 0.985 | 0.95 | 0.95 | 0.93 | 0.545 |
| JavaScript | 0.71 | 0.934 | 0.96 | 0.98 | 0.93 | 0.97 | 0.965 | 0.73 |
| JSON | 0.81 | 0.944 | 1.00 | 1.00 | 0.944 | 0.96 | 1.00 | 0.814 |
| Makefile | 0.62 | 0.92 | 0.98 | 0.98 | 0.98 | 0.92 | 0.94 | 0.56 |
| Markdown | 0.54 | 0.82 | 0.90 | 0.94 | 0.88 | 0.90 | 0.88 | 0.66 |
| Python | 0.69 | 0.928 | 0.933 | 0.969 | 0.897 | 0.91 | 0.933 | 0.626 |
| reStructuredText | 0.52 | 0.88 | 0.86 | 0.92 | 0.86 | 0.88 | 0.90 | 0.64 |
| Rust | 0.48 | 0.90 | 0.90 | 0.94 | 0.82 | 0.90 | 0.90 | 0.46 |
| Shell | 0.58 | 0.92 | 0.98 | 0.98 | 0.92 | 0.98 | 0.92 | 0.66 |
| SQL | 0.60 | 0.84 | 0.94 | 1.00 | 0.92 | 0.96 | 0.96 | 0.60 |
| Text | 0.54 | 0.90 | 0.88 | 0.90 | 0.86 | 1.00 | 0.96 | 0.64 |
| TypeScript | 0.62 | 0.96 | 0.92 | 0.96 | 0.94 | 0.88 | 0.98 | 0.48 |
| XML | 0.76 | 0.86 | 1.00 | 0.98 | 0.78 | 0.98 | 0.98 | 0.62 |
| YAML | 0.54 | 0.92 | 0.96 | 0.98 | 0.90 | 0.96 | 0.98 | 0.76 |
| **平均** | **0.626** | **0.916** | **0.948** | **0.971** | **0.921** | **0.938** | **0.944** | **0.620** |

经过任务专训后，4B 模型可以显著缩小与大模型的差距，并明显超过未训练基座模型和 Fast-Apply-7B。

## 错误分析

在主测试集中，错误主要集中在 `OUT_OF_PATCH_SIDE_EFFECT` 与 `PATCH_INCOMPLETE`：前者表示模型修改了补丁之外的内容，后者表示只应用了部分修改。

| 错误类型 | DeepSeek-V3.2 | Qwen3-4B | aiXapply-RL | Qwen3.5-397B | Kimi-K2.5 | GLM-5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `OUTPUT_INVALID` | 10.22% | 12.35% | 4.76% | 3.53% | 10.64% | 30.00% |
| `PATCH_NOT_APPLIED` | 5.11% | 8.29% | 9.52% | 9.41% | 0.00% | 0.77% |
| `PATCH_INCOMPLETE` | 18.98% | 12.87% | 27.61% | 22.35% | 21.28% | 9.23% |
| `PATCH_INCORRECT` | 8.03% | 4.06% | 11.43% | 7.06% | 4.26% | 2.31% |
| `WRONG_POSITION` | 3.65% | 5.64% | 10.48% | 11.76% | 6.38% | 2.31% |
| `OUT_OF_PATCH_SIDE_EFFECT` | 54.01% | 56.79% | 36.19% | 45.88% | 57.45% | 55.38% |

Apply 任务要求严格：即使 patch 外代码存在可改进空间，也必须保持不变。训练能显著降低副作用和无效输出，使模型行为更符合 IDE 审查场景。

## 泛化测试

| 数据集 | Qwen3-4B | DeepSeek-V3.2 | aiXapply-RL | aiXapply-SFT |
| --- | ---: | ---: | ---: | ---: |
| 主测试集 | 0.626 | 0.916 | 0.9382 | 0.944 |
| Random Placeholders | 0.696 | 0.932 | 0.948 | 0.951 |
| Chunk File | 0.5247 | 0.875 | 0.886 | 0.900 |
| Untrained Languages | 0.6399 | 0.932 | 0.938 | 0.941 |
| Long Context | 0.2353 | 0.588 | 0.6471 | 0.843 |

SFT 在长上下文场景中明显更强，可能是因为每个输出 token 都有直接监督，更有利于学习完整文件保持能力。RL 在多数场景中仍然具有竞争力，并在 Diff-XYZ 中表现出更强的跨格式泛化。

### 未训练语言

| 模型 | C# | CSS | PHP | SystemVerilog | 平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B | 0.6356 | 0.623 | 0.6409 | 0.6695 | 0.6399 |
| DeepSeek-V3.2 | 0.889 | 0.963 | 0.927 | 0.932 | 0.932 |
| aiXapply-RL | 0.890 | 0.958 | 0.936 | 0.958 | 0.938 |
| aiXapply-SFT | 0.924 | 0.937 | 0.946 | 0.958 | 0.941 |

## RQ3：推理效率

全文 Apply 的输出具有明显“高复用、强拷写”特征：大部分输出 token 都来自原始文件，因此非常适合 speculative decoding。

### Speculative Decoding 对比

![Speculative decoding 延迟总览](../assets/figures/aiXapply-speculative-decoding-overview.png)

*图 2：Baseline、suffix decoding 与 n-gram speculative decoding 在平均延迟、P95 延迟和吞吐上的对比。*

| 方法 | 配置 | 平均延迟 | P95 延迟 | 吞吐 | 平均加速比 |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | 无 speculative decoding | 28.831s | 90.230s | 102.04 tok/s | 1x |
| Suffix 默认 | `k=32, depth=16` | 5.751s | 20.741s | 509.54 tok/s | 5.01x |
| N-gram 默认 | `n=16, k=32` | 2.168s | 6.935s | 1343.99 tok/s | 13.3x |
| Suffix 最优 | `k=32, depth=32` | 2.842s | 10.062s | 1028.79 tok/s | 10.14x |
| N-gram 最优 | `n=7, k=128` | 1.06s | 3.381s | 2692.01 tok/s | 27.19x |

`n-gram` 更适合 aiXapply，因为它直接利用 prompt / source file 中的重复片段。最终推荐配置为 **`n=7, k=128`**。

### N-gram 参数搜索

![N-gram 参数消融热力图](../assets/figures/aiXapply-ngram-ablation-heatmap.png)

*图 3：N-gram 平均延迟热力图。增大 `k` 带来明显收益，`n=7, k=128` 是当前搜索范围内的最优配置。*

固定 `k=32` 时，`n=7~12` 进入稳定高性能区间：

| 配置 | 平均延迟 | P95 延迟 | 吞吐 |
| --- | ---: | ---: | ---: |
| `n=1, k=32` | 9.901s | 41.749s | 296.74 tok/s |
| `n=3, k=32` | 2.938s | 11.849s | 994.75 tok/s |
| `n=7, k=32` | 2.255s | 7.658s | 1294.09 tok/s |
| `n=12, k=32` | 2.137s | 6.785s | 1365.04 tok/s |
| `n=16, k=32` | 2.177s | 6.982s | 1340.56 tok/s |

固定 `n=7` 时，增大 `k` 能持续改善延迟，直到当前搜索范围内的最优点：

| 配置 | 平均延迟 | P95 延迟 | 吞吐 |
| --- | ---: | ---: | ---: |
| `n=7, k=16` | 3.799s | 13.399s | 771.09 tok/s |
| `n=7, k=48` | 1.724s | 5.693s | 1688.80 tok/s |
| `n=7, k=64` | 1.443s | 4.793s | 2012.70 tok/s |
| `n=7, k=96` | 1.160s | 4.009s | 2492.26 tok/s |
| `n=7, k=128` | 1.092s | 3.328s | 2649.89 tok/s |

### Suffix 参数搜索

![Suffix decoding 参数消融](../assets/figures/aiXapply-suffix-ablation.png)

*图 4：Suffix decoding 参数消融。`depth` 是最关键参数，`k` 和 `min_token_prob` 的影响较弱。*

Suffix decoding 也能利用 Apply 任务中的复写结构，但最优效果仍弱于 n-gram speculation。

在 `k` 消融中，suffix decoding 对 speculative token 数量并不敏感：

| 配置 | 平均延迟 | P95 延迟 | 吞吐 |
| --- | ---: | ---: | ---: |
| `k=16` | 5.735s | 20.660s | 511.57 tok/s |
| `k=24` | 5.738s | 20.707s | 511.26 tok/s |
| `k=32` | 5.736s | 20.690s | 511.25 tok/s |
| `k=48` | 5.846s | 21.032s | 501.88 tok/s |
| `k=64` | 6.006s | 21.532s | 488.36 tok/s |

`depth` 是 suffix 中最关键的参数：

| 配置 | 平均延迟 | P95 延迟 | 吞吐 |
| --- | ---: | ---: | ---: |
| `depth=8` | 11.212s | 39.973s | 262.07 tok/s |
| `depth=16` | 5.725s | 20.708s | 512.12 tok/s |
| `depth=24` | 3.699s | 13.159s | 791.93 tok/s |
| `depth=32` | 2.842s | 10.062s | 1028.79 tok/s |

`max_spec_factor` 有一定正向收益，但影响小于 `depth`：

| 配置 | 平均延迟 | P95 延迟 | 吞吐 |
| --- | ---: | ---: | ---: |
| `factor=1.0` | 6.551s | 23.743s | 448.04 tok/s |
| `factor=1.5` | 6.082s | 21.923s | 482.52 tok/s |
| `factor=2.0` | 5.735s | 20.731s | 511.40 tok/s |
| `factor=3.0` | 5.245s | 18.756s | 559.08 tok/s |

`min_token_prob` 在本任务中几乎没有影响：

| 配置 | 平均延迟 | P95 延迟 | 吞吐 |
| --- | ---: | ---: | ---: |
| `min_prob=0.00` | 5.727s | 20.723s | 512.17 tok/s |
| `min_prob=0.05` | 5.736s | 20.694s | 511.29 tok/s |
| `min_prob=0.10` | 5.743s | 20.726s | 510.78 tok/s |

### 按语言稳定性

![按语言的推理效率](../assets/figures/aiXapply-language-efficiency.png)

*图 5：最优 n-gram 配置下，不同语言的平均延迟和吞吐表现。*

最优配置在全量 1,637 条测试集上的分语言表现：

| 语言 | 样本数 | 平均延迟 | P95 延迟 | 平均输出 tokens | 吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 1,637 | 1.06s | 3.381s | 2853.4 | 2692.01 tok/s |
| C | 130 | 1.304s | 3.533s | 3832.4 | 2888.11 tok/s |
| C++ | 128 | 1.493s | 4.370s | 4076.1 | 2686.26 tok/s |
| Java | 200 | 1.058s | 3.364s | 2666.1 | 2469.23 tok/s |
| JavaScript | 200 | 0.866s | 2.603s | 2464.1 | 2791.22 tok/s |
| Python | 195 | 1.118s | 3.568s | 3259.7 | 2863.29 tok/s |
| SQL | 50 | 1.868s | 5.687s | 4231.5 | 2240.44 tok/s |
| XML | 50 | 1.805s | 5.818s | 3959.8 | 2168.34 tok/s |

所有语言组均成功完成。不同语言间的性能差异主要来自输出长度和文件结构，而不是某些语言上策略失效。

### 按上下文长度稳定性

![按上下文长度和修改幅度的推理效率](../assets/figures/aiXapply-context-edit-efficiency.png)

*图 6：按 source context length 和 edit magnitude 分组的延迟结果。长上下文和轻量修改通常需要生成更多复制 token，因此绝对延迟更高。*

完整测试集按 source token 数切为三档：

- `short`: `source_tokens <= 1077.6667`
- `medium`: `1077.6667 < source_tokens <= 2808.6667`
- `long`: `source_tokens > 2808.6667`

| 档位 | 样本数 | 平均延迟 | P95 延迟 | 平均输出 tokens | 吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 1,637 | 1.06s | 3.381s | 2853.4 | 2692.01 tok/s |
| Short | 546 | 0.383s | 0.491s | 724.0 | 1846.52 tok/s |
| Medium | 545 | 0.670s | 0.996s | 1860.1 | 2698.99 tok/s |
| Long | 546 | 2.126s | 4.715s | 5974.4 | 2768.57 tok/s |

长文件会提高绝对延迟，因为模型需要生成更多 token；但 medium 与 long 组吞吐仍保持在约 `2700 tok/s`，说明 n-gram speculation 在长上下文场景下依然有效。

### 按修改幅度稳定性

完整测试集也按 edit ratio 切为三档：

- `light`: `edit_ratio <= 0.053937`
- `medium`: `0.053937 < edit_ratio <= 0.135440`
- `heavy`: `edit_ratio > 0.135440`

| 档位 | 样本数 | 平均延迟 | P95 延迟 | 平均输出 tokens | 吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 1,637 | 1.06s | 3.381s | 2853.4 | 2692.01 tok/s |
| Light | 546 | 1.665s | 4.242s | 4848.7 | 2866.08 tok/s |
| Medium | 545 | 0.820s | 2.377s | 2175.8 | 2604.06 tok/s |
| Heavy | 546 | 0.694s | 1.941s | 1534.6 | 2169.77 tok/s |

轻微修改并不一定最快：轻微修改往往意味着需要复写更长的原文件，因此平均输出 token 最高。三组吞吐均维持在较高水平，说明最优 n-gram 配置在低修改比例和高修改比例样本上都比较稳定。

## RQ4：SFT vs RL

SFT 与 RL 具有互补优势：

- **SFT** 更适合主分布 Full-File Apply 准确率和长上下文保持。
- **RL** 更有利于局部性控制和跨编辑格式泛化，尤其体现在 Diff-XYZ 上。

### Diff-XYZ 跨格式泛化

| 模型 | Overall | Java | JavaScript | Kotlin | Python | Rust |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B | 0.596 | 0.630 | 0.610 | 0.620 | 0.590 | 0.530 |
| aiXapply-SFT | 0.479 | 0.510 | 0.470 | 0.490 | 0.425 | 0.500 |
| aiXapply-RL | 0.664 | 0.680 | 0.715 | 0.655 | 0.645 | 0.625 |

即使没有显式用 diff 格式训练，RL 模型在 Diff-XYZ 上仍优于基座模型；SFT 则更强地专门化到 Full-File Apply prompt 格式。

## 外部迁移：Fast-Apply 测试集

Fast-Apply 测试集可作为外部压力测试，但其中存在一些对严格 Full-File Apply 不够友好的歧义或噪声样本。人工抽查错误样本发现，部分样本存在参考答案不唯一、patch 锚点不足或 patch 本身质量问题。

| 模型 | 准确率 |
| --- | ---: |
| DeepSeek3.2 | 0.66 |
| Fast-Apply | 0.46 |
| aiXapply-RL | 0.525 |
| aiXapply-SFT | 0.52 |

因此，这组结果更适合作为外部有效性压力测试，而不是 aiXapply 的主 leaderboard。

## 关键结论

1. 在测试过的编辑表示中，全文 Apply 具有最高的一次性融合准确率。
2. 专用 4B 模型可以在 Apply 任务上接近甚至超过许多更大模型。
3. 主要错误模式来自补丁外副作用和补丁应用不完整。
4. 由于 Apply 输出具有大量复写特征，n-gram speculative decoding 非常有效。
5. SFT 与 RL 各有优势：SFT 更擅长保持长文件结构，RL 更擅长迁移到不同编辑格式。
