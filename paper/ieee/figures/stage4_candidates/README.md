# Stage 4 Candidate Figures

这些图均由真实实验数据生成，暂时不直接加入正文，供后续挑选。

## 数据来源

- `experiments/results/plan_b_stage4/exp_stage4_fair_eval_3000_conservative_and_20260601/fixed_threshold_metrics_with_conservative_and.csv`
- `experiments/results/plan_b_stage4/icdm_revision/summary_20260530/llava_coco_caption_val2014_5k_ckpt1500_20260601.csv`
- `experiments/results/plan_b_stage4/exp_stage4_fair_eval_3000_bootstrap_ci_20260531/metrics.json`

## 候选图

- `01_pair_precision_recall_f1.svg`: 主结果图候选：展示 pair-level P/R/F1，突出 Stage 4 joint 的平衡优势。
- `02_pair_f1_confidence_intervals.svg`: 主结果图候选：展示 F1 和 95% CI，强调 joint 与 baselines 的稳定差距。
- `03_precision_recall_tradeoff.svg`: tradeoff 图候选：展示 precision-recall 平面，bubble size 表示预测阳性率。
- `04_error_decomposition_tp_fp_fn.svg`: 错误分解图候选：展示 TP/FP/FN，突出 naive union 的 false positive 问题。
- `05_f1_delta_bootstrap_ci.svg`: 显著性/稳健性图候选：展示 joint 相对 naive union/image-only 的 F1 delta 和 bootstrap CI。
- `06_coco_retention_caption_metrics.svg`: downstream 图候选：展示 kept pairs 与 CIDEr/BLEU-4 的 tradeoff。
- `07_dedup_rate_vs_cider.svg`: downstream tradeoff 图候选：展示 dedup rate 与 CIDEr，突出 E 的保守保留点。
- `08_kept_pairs_vs_caption_metrics.svg`: downstream 多指标图候选：展示 kept pairs 与 CIDEr/BLEU-4/ROUGE-L。
- `paper_style/01_pair_precision_recall_f1_paper.svg`: paper-style 版本：更接近原文 Matplotlib/IEEE 图风格。
- `paper_style/02_pair_f1_confidence_intervals_paper.svg`: paper-style 版本：更接近原文 Matplotlib/IEEE 图风格。
- `paper_style/03_precision_recall_tradeoff_paper.svg`: paper-style 版本：更接近原文 Matplotlib/IEEE 图风格。
- `paper_style/04_error_decomposition_tp_fp_fn_paper.svg`: paper-style 版本：更接近原文 Matplotlib/IEEE 图风格。
- `paper_style/05_f1_delta_bootstrap_ci_paper.svg`: paper-style 版本：更接近原文 Matplotlib/IEEE 图风格。
- `paper_style/06_coco_retention_caption_metrics_paper.svg`: paper-style 版本：更接近原文 Matplotlib/IEEE 图风格。
- `paper_style/07_dedup_rate_vs_cider_paper.svg`: paper-style 版本：更接近原文 Matplotlib/IEEE 图风格。
