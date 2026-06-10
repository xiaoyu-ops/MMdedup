# MMdedup Plan B Stage 4 最终结果报告

生成时间: 2026-05-23 01:32:00 Asia/Shanghai

## 0. 摘要

- 目标：服务 CIKM 2026 Full Paper 的 Plan B 修订。
- 核心回应：新增 Stage 4，使用 CLIP joint embedding 做图文对级别跨模态去重。
- 当前安全结论：在 1,000 条 hard-candidate 标注集上，Stage 4 优于 text-only 和 naive union，但 image-only 的 F1 仍更高。
- 当前下游状态：A/B/C/D 的 25K/2000-step LLaVA LoRA 已完成；E rerun 与 VQAv2/TextVQA 评测仍待完成，不能提前填数。
- 当前补齐项：Stage 4 joint 阈值扫描展开表与现有可报告效率表已加入。
- 本报告不包含进度面板逐项对照，只列可直接用于论文写作的数据与 source-of-truth。

## 1. 各实验实现思路

### 1.1 Stage 4 跨模态图文对去重

- 输入带 pair id、image path、caption text 的图文对。
- 使用 CLIP/OpenCLIP image encoder 与 text encoder 分别得到 e_img 和 e_txt。
- 主方案使用 concat([e_img; e_txt]) 作为 joint embedding；weighted sum 作为可选消融。
- joint similarity 超过 tau_cross 时判为重复图文对。
- 保留 CLIP 图文对齐分数更高的 pair；打平时再用图像质量或 manifest 顺序。

### 1.2 CC3M hard-candidate ground truth 构建

- 不从 CC3M 随机抽 pair-pairs，因为自然重复比例太低。
- 先构建 200K CC3M 图文对池，再用 image/text/joint 相似度挖掘 500K candidate edges。
- 抽 1,000 条 hard candidates 人工标注为 duplicate、near-duplicate、not-duplicate。
- 评价时 positive class = duplicate + near-duplicate。

### 1.3 Stage 4 主评价

- Baselines 包括 image-only、text-only、image/text independent drops 的 naive union。
- Ours 使用 joint similarity threshold 的 Stage 4 joint pair dedup。
- 指标包括 precision、recall、F1、TP/FP/TN/FN 与 threshold sweep。

### 1.4 LLaVA 下游验证

- 构建 A raw、B image-only、C text-only、D naive union、E Stage 4 joint 五组训练数据。
- 在单张 RTX 3090 上用 LoRA/QLoRA 微调 LLaVA-1.5-7B，五组超参保持一致。
- 优先评测 VQAv2；TextVQA 视时间补充。
- training loss 不能当作下游性能。

### 1.5 效率与消融

- 效率表记录 CLIP embedding time、candidate search/clustering time、wall-clock time、GPU peak memory 与 throughput。
- 阈值敏感性报告 image/text/joint/naive threshold 下的 dedup rate 与 P/R/F1。
- 完整消融在下游评测完成后覆盖 raw、单模态、naive union 与 Stage 4。


## 2. 完整数据清单

### 表 1. 方案所需数据清单

| 数据项 | 当前值 | 论文用途 | Source-of-truth |
| --- | --- | --- | --- |
| CC3M pool | 200,000 | 数据构建与训练池 | data audit |
| Candidate pair-pairs | 500,000 | hard-candidate mining | data audit |
| High-joint candidates | 129,139 | 标注候选池 | data audit |
| Annotated pair-pairs | 1,000 | Stage 4 GT benchmark | adjudication metrics |
| Positive labels | 295 | duplicate + near-duplicate | data audit |
| Negative labels | 705 | not-duplicate | data audit |

Note: 标注集来自 mined hard candidates，不能解释为原始 CC3M 的自然重复率。

### 表 2. 人工标注与 Audit 状态

| 指标 | 数值 | 解释 |
| --- | --- | --- |
| duplicate | 58 | 严格重复 |
| near-duplicate | 237 | 语义或视觉近重复 |
| not-duplicate | 705 | 负类 |
| audit rows | 200 | 当前内部 audit 行数 |
| agreement rate | 暂不报告 | audit label 当前默认等于 primary label，不是真实合作者复核 |

Note: 真实合作者一致性统计尚未完成，论文不能声称 agreement_rate=1.0。

### 表 3. 1,000 条标注候选对上的 Stage 4 主评价

| 方法 | 阈值 | Precision | Recall | F1 | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image-only | 0.8 | 61.92% | 63.39% | 0.6265 | 187 | 115 | 590 | 108 |
| Text-only | 0.6 | 29.50% | 100.00% | 0.4556 | 295 | 705 | 0 | 0 |
| Naive union | 0.6 | 29.50% | 100.00% | 0.4556 | 295 | 705 | 0 | 0 |
| Stage 4 joint | 0.85 | 51.02% | 68.14% | 0.5835 | 201 | 193 | 512 | 94 |

Note: Stage 4 joint 优于 text-only/naive union，但当前 image-only 的 F1 最高，写作时必须如实说明。

### 表 4. 误差分析摘要

| 指标 | 数值 | 论文中如何使用 |
| --- | --- | --- |
| Stage 4 false positives | 193 | 分析过删风险 |
| Stage 4 false negatives | 94 | 分析漏检风险 |
| Image correct / joint wrong | 142 | 解释 image-only 当前更强的原因 |
| Joint correct / image wrong | 78 | 说明 Stage 4 的增量价值 |
| Joint FP with identical captions | 82 (42.49%) | caption-template failure mode |

### 表 5. 200K 图文池上的 A/B/C/D/E 训练数据规模

| 配置 | 名称 | 原始样本 | 保留 | 删除 | 去重率 | 规则 |
| --- | --- | --- | --- | --- | --- | --- |
| A | raw | 200,000 | 200,000 | 0 | 0.00% | n/a |
| B | image_only | 200,000 | 171,997 | 28,003 | 14.00% | image>=0.8 |
| C | text_only | 200,000 | 104,395 | 95,605 | 47.80% | text>=0.6 |
| D | naive_union | 200,000 | 101,752 | 98,248 | 49.12% | image>=0.8 OR text>=0.6 |
| E | stage4_joint | 200,000 | 177,957 | 22,043 | 11.02% | joint>=0.85 |

Note: 这些 split sizes 用于定义下游训练 manifest；E 是 Stage 4 joint 对照组。

### 表 6. Stage 4 joint 阈值扫描展开表

| τ_cross | Precision | Recall | F1 | 200K 去重率 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 0.6 | 29.50% | 100.00% | 0.4556 | 47.71% |  |
| 0.65 | 29.50% | 100.00% | 0.4556 | 45.08% |  |
| 0.7 | 29.50% | 100.00% | 0.4556 | 40.21% |  |
| 0.75 | 29.50% | 100.00% | 0.4556 | 32.34% |  |
| 0.8 | 29.50% | 100.00% | 0.4556 | 21.79% |  |
| 0.85 | 51.02% | 68.14% | 0.5835 | 11.02% | selected |
| 0.9 | 73.11% | 29.49% | 0.4203 | 4.61% |  |
| 0.95 | 88.64% | 13.22% | 0.2301 | 1.62% |  |

Note: 评价指标来自 1,000 条人工标注 hard candidates；去重率来自 200K CC3M mined candidate graph。

### 表 7. 当前可报告效率表

| 阶段 | 规模 | 时间 | Throughput | 硬件/备注 |
| --- | --- | --- | --- | --- |
| CC3M 200K prepare | 200,000 pairs | 41.20 min | 80.91 pairs/s | Windows data preparation; not counted as Stage 4 algorithm overhead |
| OpenCLIP candidate mining | 200,000 pairs / 500,000 edges | 1.28 h | 43.35 pairs/s | Windows RTX 3090; includes embedding/backend search pipeline |
| Threshold-to-split graph computation | 200,000 pairs / 500,000 edges | 18.35 s | 27248.20 edges/s | Mac-side source-of-truth consolidation |
| A/B/C/D/E manifest writing | 200,000 pairs | 23.25 s | 8600.69 pairs/s | Generates LLaVA train manifests |
| 1K labeled evaluation | 1,000 labeled pair-pairs | 0.01 s | 88143.41 rows/s | Metric computation only |

Note: 这是当前已有 source-of-truth 下可报告的效率版本；GPU peak memory 还需要后续补充到正式效率表。

### 表 8. LLaVA-1.5-7B LoRA 训练状态

| 配置 | 名称 | 状态 | 样本数 | 步数 | 最终 loss | Wall-clock h | Peak GB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | raw | trained | 25,000 | 2000 | 4.2365 | 5.70 | 5.525 |
| B | image-only | trained | 25,000 | 2000 | 0.4428 | 5.62 | 5.448 |
| C | text-only | trained | 25,000 | 2000 | 3.8543 | 5.72 | 5.523 |
| D | naive union | trained | 25,000 | 2000 | 3.5730 | 5.85 | 5.524 |
| E | Stage 4 joint | pending |  | 2000 target |  |  |  |

Note: Training loss 只是工程和收敛证据，不是 VQAv2/TextVQA 下游性能。

### 表 9. VQAv2 评测结果

| 配置 | Seed 1 Acc | Seed 2 Acc | Mean +/- Std | 状态 |
| --- | --- | --- | --- | --- |
| A |  |  |  | pending |
| B |  |  |  | pending |
| C |  |  |  | pending |
| D |  |  |  | pending |
| E |  |  |  | pending |

Note: 保持空白，直到 VQAv2 evaluation metrics.json 生成。

### 表 10. TextVQA 评测结果

| 配置 | Seed 1 Acc | Seed 2 Acc | Mean +/- Std | 状态 |
| --- | --- | --- | --- | --- |
| A |  |  |  | pending |
| B |  |  |  | pending |
| C |  |  |  | pending |
| D |  |  |  | pending |
| E |  |  |  | pending |

Note: 时间允许再补，不能虚构。

### 表 11. 缺失数据与后续补数位置

| 缺失项 | 用途 | 预期 Source-of-truth |
| --- | --- | --- |
| E 25K/2000-step metrics | 补全表 6 | Windows E rerun metrics.json mirrored to windows_sync |
| VQAv2 A/B/C/D/E metrics | 补全表 7 与下游 claim | exp_llava_stage4_vqa_vqav2_quick_* metrics.json |
| TextVQA A/B/C/D/E metrics | 补全表 8，可选 | TextVQA eval metrics.json |
| 真实合作者 audit labels | inter-annotator agreement | adjudicated annotation CSV with collaborator labels |
| 完整效率计时 | system overhead table | Stage 4 embedding/search/runtime metrics on Windows RTX 3090 |
| SSCD baseline | 回应强 image baseline | SSCD eval metrics on CC3M GT / image benchmarks |


## 3. Source-of-truth 文件

- `experiments/results/plan_b_stage4/experiment_ledger.csv`
- `experiments/results/plan_b_stage4/data_audits/2026-05-20_data_reasonableness_audit.json`
- `experiments/results/plan_b_stage4/exp_stage4_eval_1000_200k_high_joint_20260519/metrics.json`
- `experiments/results/plan_b_stage4/exp_stage4_eval_1000_200k_high_joint_20260519/per_threshold_metrics.csv`
- `experiments/results/plan_b_stage4/exp_stage4_training_manifests_200k_20260520/abcde_split_sizes.csv`
- `experiments/results/plan_b_stage4/windows_sync/exp_llava_stage4_train25k_*_25000_2000steps_20260521/metrics.json`
- Pending E: Windows 原始实验目录完成后同步到 `experiments/results/plan_b_stage4/windows_sync/`。
