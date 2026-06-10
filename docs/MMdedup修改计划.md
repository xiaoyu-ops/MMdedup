## MMdedup 论文修改计划

### 一、原文存在的核心问题

#### 问题 1：所谓”多模态”实质上是三个单模态的拼接

原稿提出图像、音频、文本三个独立去重模块，但模态之间没有任何交互。这是 R2/R3 给出 reject 的最深层原因。真正的多模态必须体现模态间的关联，例如图文对（image-caption pair）作为整体去重，而不是分别处理图像和文字。

#### 问题 2：合成数据评估不被认可

原稿使用 ImageNet-Expanded、Amazon-Expanded、ESC-Expanded 三个合成数据集，通过人工注入重复构造 ground truth。R2 直接指出这种”自问自答”评估方式不被接受。

#### 问题 3：标题声称 MLLM 训练但无 MLLM 实验

论文标题是 “for MLLM Training”，但全文没有任何在 MLLM 上的实际验证，是 claim 与实验的严重脱节。

#### 问题 4：数字一致性问题

摘要、Introduction、Tables 之间存在数字矛盾。R1 weak reject 的核心扣分点。

#### 问题 5：图像端 baseline 不够强

原稿用 SimCLR 作为图像深度学习 baseline，但 SimCLR 是对比学习方法，不是为近重复检测设计。SSCD（CVPR 2022）是图像近重复检测的 SOTA baseline，必须加入。

#### 问题 6：论文中存在自我否定表述

Section 2.6 的 “Rather than proposing novel algorithms for individual components, our contribution lies in their systematic integration…” 直接把自己定位为”没有算法贡献”。R1 直接引用此句质疑创新性。

#### 问题 7：不利结果未做正面讨论

表 5 中 pHash 的 Precision 和 Recall 都高于本文 CLIP 方法，但正文没有解释。审稿人会当作”避重就轻”。表 6 中 MD5 在某些指标上的优势同样未讨论。

#### 问题 8：“首次为 MLLM 训练做多模态去重”的 claim 立不住

SemDeDup（在 LAION 上验证 CLIP 训练效果）、DataComp、FairDeDup 都做过相关工作，原稿没有充分讨论。

### 二、解决思路

| 问题 | 解决方案 |
| --- | --- |
| 问题 1：三个单模态拼接 | 新增 Stage 4：跨模态去重模块，使用 CLIP joint embedding 在图文对级别识别重复 |
| 问题 2：合成数据 | 使用 CC3M 真实数据集；人工标注 ≥ 1000 对真实重复作为 ground truth |
| 问题 3：无 MLLM 验证 | 在 LLaVA-1.5-7B 上做 LoRA fine-tune 实验，对比不同去重配置在 VQAv2 / TextVQA 上的下游性能 |
| 问题 4：数字不一致 | 建立 source-of-truth 表，所有论文数字逐项追溯实验日志 |
| 问题 5：baseline 不够强 | 加入 SSCD 作为图像深度学习 baseline；SimCLR 保留作参考但不作主对比 |
| 问题 6：自我否定表述 | 改写 Section 2.6，强调跨模态去重的算法贡献 |
| 问题 7：不利结果未讨论 | 在正文加段正面讨论 pHash / MD5 / MFCC 的边界条件 |
| 问题 8：claim 不准确 | claim 改为”首次在 MLLM 训练流水线中端到端验证图文对级别的跨模态去重”；Related Work 加深 SemDeDup / FairDeDup / DataComp 对比 |

### 三、实验设计

#### 实验 1：跨模态去重模块（Stage 4）

目的：实现新的跨模态去重模块，作为本次修改的核心 novelty。

方法： 对每个图文对 (I_i, T_i)：

用 CLIP-ViT-B/16 分别编码图像和文本，得到 e_img ∈ R^512 和 e_txt ∈ R^512（L2 归一化）

构造 joint embedding，对比两种方案：

方案 A：concat —— e_joint = [e_img; e_txt] ∈ R^1024

方案 B：加权和 —— e_joint = α · e_img + (1−α) · e_txt ∈ R^512，扫描 α ∈ {0.3, 0.5, 0.7}

在 joint embedding 空间用 spherical k-means 聚类（沿用现有 image stage 的 Faiss 实现，k = 2000）

簇内做 pairwise cosine similarity，sim > τ_cross 视为跨模态重复

重复对中保留 quality 较高的一个

数据集：1000 对人工标注 CC3M ground truth（见实验 2）

工具：open_clip（CLIP-ViT-B/16）、Faiss-GPU、scikit-learn

需要收集的数据：

| 数据项 | 字段 | 说明 |
| --- | --- | --- |
| Joint embedding 方式对比 | 方式, P, R, F1 | concat vs weighted sum 在 ground truth 上的对比，用于确定主方案 |
| 阈值扫描表 | τ_cross, P, R, F1, 去重率 | τ_cross ∈ {0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95} |
| 最优配置最终性能 | joint 方式, τ_cross, P, R, F1 | 论文主表数据 |
| 与 naive baseline 对比 | 方法, P, R, F1 | “图像独立去重 ∪ 文本独立去重”作为 baseline，证明跨模态联合的增量价值 |
| 计算开销 | embedding 时间, 聚类时间, 单对处理时间, GPU 显存峰值 | efficiency 分析 |

结果记录表（实验完成后填入）

表 1.1　Joint embedding 方式对比（在 1000 对 ground truth 上）

| 方式 | α | P | R | F1 | 备注 |
| --- | --- | --- | --- | --- | --- |
| concat | — |  |  |  |  |
| weighted sum | 0.3 |  |  |  |  |
| weighted sum | 0.5 |  |  |  |  |
| weighted sum | 0.7 |  |  |  |  |

表 1.2　τ_cross 阈值扫描（采用最优 joint embedding 方式）

| τ_cross | P | R | F1 | 去重率 (%) |
| --- | --- | --- | --- | --- |
| 0.60 |  |  |  |  |
| 0.65 |  |  |  |  |
| 0.70 |  |  |  |  |
| 0.75 |  |  |  |  |
| 0.80 |  |  |  |  |
| 0.85 |  |  |  |  |
| 0.90 |  |  |  |  |
| 0.95 |  |  |  |  |

表 1.3　最优配置最终性能（用于论文主表）

| Joint 方式 | τ_cross | P | R | F1 | 选定理由 |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

表 1.4　与 naive multimodal baseline 对比（核心论证）

| 方法 | P | R | F1 | 备注 |
| --- | --- | --- | --- | --- |
| 图像独立去重 ∪ 文本独立去重（naive） |  |  |  | baseline |
| Stage 4 跨模态联合去重（本文） |  |  |  | ours |
| 提升幅度（绝对值） |  |  |  | F1 至少 +0.05 才有说服力 |

表 1.5　计算开销（基于 100K 图文对测试）

| 项目 | 值 | 单位 | 备注 |
| --- | --- | --- | --- |
| Embedding 生成时间 |  | sec / 1K pairs | CLIP 推理 |
| 聚类时间 |  | sec | k=2000，Faiss-GPU |
| 单图文对处理时间 |  | ms | end-to-end |
| GPU 显存峰值 |  | GB |  |
| Wall-clock 总耗时 |  | min | 100K pairs 全流程 |

关键判定：跨模态去重在 ground truth 上 F1 ≥ 0.6 才算成功。F1 < 0.5 必须暂停讨论是否换 BLIP / SigLIP 或调整算法。

#### 实验 2：CC3M 真实数据 Ground Truth 标注

目的：替换合成数据评估，回应 R2 的核心质疑。

方法：

从 CC3M validation split 下载图文对（必要时扩展到 train split），目标 100K – 300K 对

随机采样 5000 对图文对（必须是配对，不是单独的图或文本）

双盲标注：至少 2 名标注人独立判断每对的关系，三类标签：

duplicate：图文都重复

near-duplicate：语义重复但有改写或视觉变换

not-duplicate：不重复

分歧样本由第三人仲裁

计算 Cohen’s kappa；< 0.6 必须重新对齐标注规范再做一轮

最终目标：≥ 1000 对带标签数据，其中 duplicate + near-duplicate ≥ 200 对

工具：Label Studio 或自建简单 web 标注工具

需要收集的数据：

| 数据项 | 字段 | 说明 |
| --- | --- | --- |
| 标注主表 | file_id_1, file_id_2, 标注人 ID, 标签, 标注时间戳 | 主交付物 |
| 仲裁记录 | file_id_1, file_id_2, 仲裁人, 最终标签, 仲裁原因 | 用于审稿人质疑标注质量时回应 |
| 一致性指标 | Cohen’s kappa, Fleiss’ kappa | 论文中报告标注质量 |
| 标签分布 | 三类各自数量 | 确认正负样本比例合理 |
| 标注耗时 | 平均每对秒数 | 估算后续扩展成本 |

结果记录表（标注完成后填入）

表 2.1　数据集来源与采样

| 项目 | 值 | 备注 |
| --- | --- | --- |
| 数据来源 |  | CC3M val / train split |
| 实际下载图文对总数 |  | 用于采样池 |
| 链接失效率 | % |  |
| 采样总对数 |  | 目标 5000 |

表 2.2　标注汇总

| 指标 | 值 |
| --- | --- |
| 完成标注的样本对数 |  |
| 标注人数 |  |
| 仲裁的样本对数 |  |
| 平均每对标注耗时（秒） |  |

表 2.3　标签分布

| 标签 | 数量 | 占比 (%) |
| --- | --- | --- |
| duplicate |  |  |
| near-duplicate |  |  |
| not-duplicate |  |  |
| 合计 |  | 100 |

表 2.4　标注一致性指标

| 指标 | 值 | 是否 ≥ 0.6 |
| --- | --- | --- |
| Cohen’s kappa（标注人 1 vs 2） |  |  |
| Fleiss’ kappa（≥ 3 人参与时） |  |  |

重要说明：标注主表（5000 行 CSV，含每对的 file_id_1、file_id_2、标注人 ID、各自标签、最终标签、仲裁记录）作为 supplementary material 提交，不放论文正文。

#### 实验 3：SSCD Baseline 补充

目的：回应审稿人对图像 baseline 不够强的意见。

方法：

从 Meta Research 仓库下载 SSCD 公开权重（推荐 sscd_disc_mixup checkpoint）

在 ImageNet-Expanded（沿用原稿）和 CC3M ground truth 上分别跑 SSCD 推理

用相同的 cosine similarity 阈值策略做去重，扫描阈值

与原表 5 中所有方法在同一指标上对比

工具：SSCD 官方仓库（PyTorch），用其提供的预处理 pipeline

需要收集的数据：

| 数据项 | 字段 | 说明 |
| --- | --- | --- |
| SSCD 在 ImageNet-Expanded 上指标 | 阈值, Dedup 率, P, R, F1, 下游 Acc | 加入更新后的 Table 5 |
| SSCD 在 CC3M ground truth 上指标 | 阈值, P, R, F1 | 新表，与跨模态 stage 对比 |
| SSCD 推理速度 | imgs/sec | 与 CLIP-based 方法对比 |
| SSCD GPU 显存占用 | GB | 与本文方法对比 |

结果记录表（实验完成后填入）

表 3.1　SSCD 在 ImageNet-Expanded 上的阈值扫描

| 阈值 | Dedup 率 (%) | P (%) | R (%) | F1 | 下游 Acc (%) |
| --- | --- | --- | --- | --- | --- |
| 0.60 |  |  |  |  |  |
| 0.70 |  |  |  |  |  |
| 0.80 |  |  |  |  |  |
| 0.85 |  |  |  |  |  |
| 0.90 |  |  |  |  |  |
| 0.95 |  |  |  |  |  |

表 3.2　SSCD 在 CC3M Ground Truth 上的阈值扫描

| 阈值 | P (%) | R (%) | F1 |
| --- | --- | --- | --- |
| 0.60 |  |  |  |
| 0.70 |  |  |  |
| 0.80 |  |  |  |
| 0.85 |  |  |  |
| 0.90 |  |  |  |
| 0.95 |  |  |  |

表 3.3　SSCD 与原稿方法在 ImageNet-Expanded 上的对比（更新后的 Table 5）

| 方法 | Dedup (%) | Prec (%) | Rec (%) | TP (img/s) | GPU (GB) | Acc (%) | Time (h) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No Dedup. | 0.00 | – | – | – | – | 79.94 | – |
| MD5 | 0.30 | 100.00 | 0.00 | 2640.2 | 0 | 80.10 | 6.76 |
| pHash | 60.39 | 100.00 | 99.97 | 356.2 | 0 | 72.26 | 6.33 |
| SimCLR | 69.17 | 18.26 | 99.68 | 45.0 | 21.3 | 67.80 | 11.8 |
| SemDeDup | 69.21 | 93.70 | 96.20 | 27.9 | 4.0 | 67.86 | 12.0 |
| SSCD（新增） |  |  |  |  |  |  |  |
| Ours (CLIP) | 67.25 | 85.24 | 69.25 | 101.2 | 4.6 | 69.58 | 10.5 |

#### 实验 4：MLLM 下游训练验证（最重要的实验）

目的：验证去重对 MLLM 训练的实际效果，回应”标题为 MLLM 但无验证”的核心质疑。

方法：

第一步：在 CC3M 子集上准备 5 组去重配置的训练数据：

| 配置 | 描述 | 用途 |
| --- | --- | --- |
| A | 原始数据，不去重 | baseline |
| B | 仅图像去重 | 单模态消融 |
| C | 仅文本去重 | 单模态消融 |
| D | 图像 + 文本独立去重的并集 | naive 多模态去重 baseline |
| E | 跨模态联合去重（Stage 4） | 本文方法 |

第二步：5 份训练数据各 fine-tune LLaVA-1.5-7B（LoRA），训练超参固定（batch size、learning rate、epochs、random seed）。每组跑 2 次（不同 seed），结果取均值 ± std，避免单次结果偶然性。

第三步：训练完成后在 VQAv2、TextVQA 上 zero-shot 评测。

工具：LLaVA 官方代码、PEFT（LoRA）、transformers、accelerate

资源：A100 40G ×1（理想）或 RTX 3090 24G ×1（最小可行配置，需 batch size = 1 + gradient accumulation）

需要收集的数据：

| 数据项 | 字段 | 说明 |
| --- | --- | --- |
| 5 组训练数据规模 | 配置, 原始样本数, 去重后样本数, 去重率 | 论文核心表 |
| 5 组训练时间 | 配置, GPU-hour, wall-clock 时间 | 论证去重的训练效率收益 |
| 5 组训练 loss 曲线 | 配置, epoch, train_loss, val_loss | 论文 Figure |
| VQAv2 评测结果 | 配置, seed, accuracy | 5 配置 × 2 seed = 10 个数字 |
| TextVQA 评测结果 | 配置, seed, accuracy | 同上 |
| 汇总性能表 | 配置, VQAv2 mean ± std, TextVQA mean ± std | 论文主表 |

结果记录表（实验完成后填入）

表 4.1　5 组训练数据规模

| 配置 | 描述 | 原始样本数 | 去重后样本数 | 去重率 (%) |
| --- | --- | --- | --- | --- |
| A | 不去重 |  | — | 0 |
| B | 仅图像去重 |  |  |  |
| C | 仅文本去重 |  |  |  |
| D | 独立去重并集 |  |  |  |
| E | 跨模态联合（Stage 4） |  |  |  |

表 4.2　5 组训练时间（GPU 型号：______）

| 配置 | GPU-hour | Wall-clock 时间 (h) | 相对 A 的训练时间 |
| --- | --- | --- | --- |
| A |  |  | 1.00 (baseline) |
| B |  |  |  |
| C |  |  |  |
| D |  |  |  |
| E |  |  |  |

表 4.3　VQAv2 评测结果（按 seed 分开记录）

| 配置 | Seed 1 Acc (%) | Seed 2 Acc (%) | Mean ± Std (%) |
| --- | --- | --- | --- |
| A |  |  |  |
| B |  |  |  |
| C |  |  |  |
| D |  |  |  |
| E |  |  |  |

表 4.4　TextVQA 评测结果（按 seed 分开记录）

| 配置 | Seed 1 Acc (%) | Seed 2 Acc (%) | Mean ± Std (%) |
| --- | --- | --- | --- |
| A |  |  |  |
| B |  |  |  |
| C |  |  |  |
| D |  |  |  |
| E |  |  |  |

表 4.5　汇总性能表（论文主表）

| 配置 | 训练样本数 | GPU-hour | VQAv2 Mean ± Std (%) | TextVQA Mean ± Std (%) |
| --- | --- | --- | --- | --- |
| A (baseline) |  |  |  |  |
| B (image only) |  |  |  |  |
| C (text only) |  |  |  |  |
| D (naive multimodal) |  |  |  |  |
| E (cross-modal, ours) |  |  |  |  |

核心对比：

E vs A：跨模态去重是否保持或提升 baseline 性能

E vs D：跨模态联合 vs 独立去重并集，证明联合处理的价值（这是 Stage 4 novelty 的核心论证）

B/C vs A：单模态去重的效果

训练时间 (E) vs (A)：去重带来的训练效率收益

#### 实验 5：阈值敏感性分析

目的：完整论证 Stage 4 设计选择，更新原稿 Figure 3。

方法：

在 CC3M 全量子集上跑 Stage 4，扫描 τ_cross

同时跑现有图像、文本、音频模块的阈值扫描（更新原 Figure 3 数据，因为数据集换了）

分析跨模态阈值与单模态阈值的关系

工具：现有 pipeline + Stage 4

需要收集的数据：

| 数据项 | 字段 | 说明 |
| --- | --- | --- |
| τ_cross vs 跨模态去重率 | τ_cross, 去重率 | 加入更新后的 Figure 3 |
| 各模态最优阈值下的去重率 | 模态, 最优阈值, 去重率 | 新表 |
| 跨模态阈值与单模态阈值的相关性 | image τ, text τ, cross τ, 联合去重率 | 散点图或相关系数 |

结果记录表（实验完成后填入）

表 5.1　各模态阈值 vs 去重率（更新后的 Figure 3 数据）

| 阈值 τ | 图像去重率 (%) | 文本去重率 (%) | 音频去重率 (%) | 跨模态去重率 (%) |
| --- | --- | --- | --- | --- |
| 0.10 |  |  |  |  |
| 0.20 |  |  |  |  |
| 0.30 |  |  |  |  |
| 0.40 |  |  |  |  |
| 0.50 |  |  |  |  |
| 0.60 |  |  |  |  |
| 0.70 |  |  |  |  |
| 0.80 |  |  |  |  |
| 0.90 |  |  |  |  |

表 5.2　各模态最优阈值与对应去重率

| 模态 | 最优阈值 τ* | 去重率 (%) | 选定依据 |
| --- | --- | --- | --- |
| 图像 |  |  |  |
| 文本 |  |  |  |
| 音频 |  |  |  |
| 跨模态 |  |  |  |

表 5.3　跨模态阈值与单模态阈值组合下的联合去重率（用于散点图 / 相关分析）

| Image τ | Text τ | Cross τ | 联合去重率 (%) |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

#### 实验 6：消融研究

目的：量化每个组件的贡献，更新原表 8。

方法：在 Mixed-Test（沿用原稿）和 CC3M ground truth 上做消融：

| 配置 | 说明 |
| --- | --- |
| Full MMdedup（含 Stage 4） | 完整方法 |
| w/o Sorter | 沿用原稿 ablation |
| w/o Near-Dedup | 沿用原稿 ablation |
| w/o Image Dedup | 沿用原稿 ablation |
| w/o Audio Dedup | 沿用原稿 ablation |
| w/o Text Dedup | 沿用原稿 ablation |
| w/o Stage 4 跨模态去重 | 新增，证明 Stage 4 的贡献 |
| Stage 4 用 concat | Stage 4 设计选择 ablation |
| Stage 4 用 weighted sum | Stage 4 设计选择 ablation |

工具：现有 pipeline + 修改后的 ablation 脚本

需要收集的数据：

| 数据项 | 字段 | 说明 |
| --- | --- | --- |
| 各 ablation 配置的去重率 | 配置, 图像去重率, 音频, 文本, 跨模态, 总体 | 更新原 Table 8 |
| 各 ablation 配置的 MLLM 下游性能 | 配置, VQAv2 acc, TextVQA acc | Table 8 新增列 |
| Stage 4 设计选择对比 | 设计, P, R, F1, 跨模态去重率 | 新增小表 |
| 处理时间 + 存储节省 | 配置, 时间, 存储节省 | 沿用原稿 Table 8 字段 |

结果记录表（实验完成后填入）

表 6.1　各 ablation 配置的去重率（更新后的 Table 8 主表，在 Mixed-Test 上）

| 配置 | 图像 (%) | 音频 (%) | 文本 (%) | 跨模态 (%) | 总体 (%) | 时间 (m) | 节省 (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full MMdedup（含 Stage 4） |  |  |  |  |  |  |  |
| w/o Sorter (Image only) |  | 0.00 | 0.00 | — |  |  |  |
| w/o Sorter (Audio only) | 0.00 |  | 0.00 | — |  |  |  |
| w/o Sorter (Text only) | 0.00 | 0.00 |  | — |  |  |  |
| w/o Sorter (Ground-truth) |  |  |  |  |  |  |  |
| w/o Near-Dedup |  |  |  | — |  |  |  |
| w/o Image Dedup | — |  |  |  |  |  |  |
| w/o Audio Dedup |  | — |  |  |  |  |  |
| w/o Text Dedup |  |  | — |  |  |  |  |
| w/o Stage 4 跨模态去重 |  |  |  | — |  |  |  |

表 6.2　各 ablation 配置的 MLLM 下游性能（在 CC3M 上）

| 配置 | VQAv2 Acc (%) | TextVQA Acc (%) | 备注 |
| --- | --- | --- | --- |
| Full MMdedup（含 Stage 4） |  |  |  |
| w/o Stage 4 跨模态去重 |  |  | 直接对比 Stage 4 增量 |
| w/o Image Dedup |  |  |  |
| w/o Text Dedup |  |  |  |
| Baseline（不去重） |  |  | 同 Table 4.5 配置 A |

表 6.3　Stage 4 设计选择对比（在 1000 对 ground truth 上）

| 设计 | α | P (%) | R (%) | F1 | 跨模态去重率 (%) | 推荐 |
| --- | --- | --- | --- | --- | --- | --- |
| concat | — |  |  |  |  |  |
| weighted sum | 0.3 |  |  |  |  |  |
| weighted sum | 0.5 |  |  |  |  |  |
| weighted sum | 0.7 |  |  |  |  |  |

### 四、统一实验记录规范

为保证 source-of-truth 表的可用性，每次实验运行建议记录以下字段：

| 字段 | 说明 |
| --- | --- |
| 实验 ID | 唯一标识，建议格式 exp{编号}{描述}{YYYYMMDD}，如 exp1_threshold_scan_20260520 |
| 数据集名称 + 版本 | 如 CC3M-100K-v1、ImageNet-Expanded-v1 |
| 代码 commit hash | 保证可复现 |
| 硬件配置 | GPU 型号 + 数量、CPU、内存 |
| 完整超参数 | 所有非默认值（建议直接保存为 yaml） |
| Wall-clock 运行时间 | 秒或小时 |
| GPU 峰值显存 | GB |
| 中间结果路径 | 例如 embedding 文件、cluster 结果 |
| 最终结果路径 | 用于论文的数字 / 图表所依据的文件 |
| 关键指标 | 论文里要引用的具体数字 |

每次实验执行后填入下表（每个实验对应一行，新实验追加新行）

| # | 实验 ID | 数据集 | Commit | 硬件 | 总耗时 | GPU 峰值 (GB) | 中间结果路径 | 最终结果路径 | 论文引用数字 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 |  |  |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |  |  |  |
| … |  |  |  |  |  |  |  |  |  |

## 五、当前实施进展（2026-05-12）

总体判断：当前已经完成 Mac 工程验证、Windows 3090 服务器迁移验证，以及 1K 真实 CC3M 小子集准备。项目已从“方案讨论 / 本机 smoke”推进到“真实 CC3M 小跑前置数据已就绪”的阶段。也就是说，Stage 4 的工程链路已经可运行，下一步可以开始真实 CC3M Stage 4 小跑。

重要说明：以下 smoke 结果仅用于工程验收和迁移验证，不作为论文结果或主表数字使用。论文可引用的数据必须来自后续真实 CC3M 标注集、完整 metrics 文件和 experiment_ledger.csv。

### 5.1 当前阶段位置

| 阶段 | 目标 | 当前状态 | 说明 |
| --- | --- | --- | --- |
| Mac 工程验证 | 在本机打通 Stage 4、候选挖掘、标注、评估工具链 | 已完成 | 所有 smoke 已通过；代码已提交到 codex/plan-b-stage4-pair-dedup。 |
| Windows Phase 1 | 代码迁移、SSH 控制、CUDA / 3090 环境验证 | 已完成 | Windows 可通过 Tailscale + SSH 控制；RTX 3090 可用。 |
| Windows Phase 2 | Windows 本地 smoke 和 open_clip CUDA smoke | 已完成 | 四个 smoke 通过；open_clip + CUDA 小样本 Stage 4 通过。 |
| Windows Phase 3 | 准备 1K-5K 真实 CC3M image-caption 小子集 | 已完成 | 已在 D:\data\cc3m_subset 准备 1,000 对真实 CC3M sidecar 数据。 |
| Windows Phase 4 | 真实 CC3M Stage 4 小跑 | 已完成 | 1K 真实 CC3M，open_clip CUDA，tau=0.95，完整产物已同步回 Mac。 |
| Windows Phase 5 | 真实 CC3M candidate mining | 已完成 | 基于 Phase 4 cache 挖出 2,663 条候选 pair-pairs。 |
| Windows Phase 6 | 第一版人工标注表 | 已完成 | 已生成 200 行 annotation_sheet，其中 40 行 needs_audit。 |
| Windows Phase 7+ | 仲裁、评估、扩大规模、LLaVA | 未开始 | 等待人工标注完成后进入评估；规模化实验可并行规划。 |

### 5.2 已完成的工程产物

| 产物 | 路径 / 名称 | 用途 |
| --- | --- | --- |
| Stage 4 核心实现 | pipelines/stage4_pair_dedup.py | 图文对级别去重，支持 open_clip、joint embedding、keep/drop/groups、embedding cache。 |
| Stage 4 运行入口 | experiments/scripts/run_stage4_pair_dedup.py | 正式运行 Stage 4，并输出 config、metrics、stdout/stderr、manifest。 |
| 候选挖掘脚本 | experiments/scripts/mine_stage4_candidates.py | 基于 image/text/joint similarity 生成 candidate pair-pairs。 |
| 标注表生成脚本 | experiments/scripts/build_annotation_sheet.py | 生成 annotation_sheet.csv，包含 label、needs_audit、audit_label 等字段。 |
| 标注仲裁脚本 | experiments/scripts/adjudicate_stage4_annotations.py | 统计 audit agreement / conflict，生成 final_label。 |
| 评估脚本 | experiments/scripts/evaluate_stage4_groundtruth.py | 对 image、text、naive_union、joint、max 做阈值扫描，输出 P/R/F1。 |
| Windows 迁移文档 | docs/plan_b_after_windows_migration.md | Windows 服务器阶段执行清单。 |
| Windows runbook | docs/plan_b_windows_server_runbook.md | Windows 上具体命令顺序。 |
| source-of-truth 骨架 | experiments/results/plan_b_stage4/ | 记录 daily log、ledger 和每次实验产物。 |
| CC3M 数据准备脚本 | experiments/scripts/prepare_cc3m_hf_subset.py | 从 HuggingFace CC3M 数据集导出 jpg/txt sidecar 子集。 |
| sidecar 校验脚本 | experiments/scripts/validate_sidecar_pairs.py | 校验图片/文本数量、manifest、配对完整性和图片可读性。 |

### 5.3 Windows 服务器验证结果

| 检查项 | 结果 | 备注 |
| --- | --- | --- |
| 远程控制 | 已完成 | Mac 通过 Tailscale + SSH 免密控制 Windows：sysu@100.105.237.38。 |
| 项目路径 | C:\Users\sysu\code\MMdedup | 分支 codex/plan-b-stage4-pair-dedup，提交 a70b383。 |
| GPU | NVIDIA GeForce RTX 3090, 24GB | nvidia-smi 正常。 |
| PyTorch / CUDA | torch 2.11.0+cu130；torch.cuda.is_available() = True | 已从 CUDA wheel 重装，避免 CPU 版 torch。 |
| Windows smoke | 全部通过 | smoke_stage4_pair_dedup / annotation_flow / evaluation / adjudication 均通过。 |
| open_clip CUDA smoke | 通过 | 4 对合成图文对：num_pairs=4, num_keepers=3, num_drops=1, dedup_rate=25%。仅工程验证，不作论文数据。 |
| 数据目录 | D:\data\cc3m_subset | 已放入 1,000 对真实 CC3M 图文 sidecar 数据。 |
| CC3M 1K 校验 | 通过 | jpg_count=1000, txt_count=1000, manifest_rows=1000, metrics_saved_pairs=1000, sampled_images_verified=100。 |
| Mac 元数据副本 | experiments/results/plan_b_stage4/windows_sync/cc3m_subset_1k_20260512/ | 已同步 manifest、prepare_metrics、prepare_failures、validation_summary。 |
| Stage 4 真实 1K 小跑 | experiments/results/plan_b_stage4/windows_sync/exp_stage4_cc3m_1k_20260513/ | num_pairs=1000, num_keepers=1000, num_drops=0, tau_cross=0.95。 |
| 候选挖掘 1K | experiments/results/plan_b_stage4/windows_sync/exp_stage4_candidates_1k_20260513/ | num_candidates=2663，signals=image/text/joint，min_similarity=0.70。 |
| 200 行标注表 | experiments/results/plan_b_stage4/windows_sync/exp_stage4_annotation_200_20260513/ | annotation_rows=200, audit_rows=40，覆盖 very_high/high/medium bucket。 |

### 5.4 下一步阶段产物

| 下一阶段 | 需要产物 | 验收标准 | 状态 |
| --- | --- | --- | --- |
| Phase 3：CC3M 小子集 | D:\data\cc3m_subset 下的 jpg/txt sidecar 文件 | 1K-5K 对；caption 非空；图片可打开；同 stem 配对 | 已完成：1K 对 |
| Phase 4：Stage 4 真实小跑 | exp_stage4_cc3m_1k_20260513/ | metrics、keepers、drops、groups、cache、stdout/stderr、manifest 齐全 | 已完成：1K 小跑 |
| Phase 5：候选挖掘 | stage4_candidate_pairs.csv | 候选数量非零；包含 image/text/joint similarity 和 bucket | 已完成：2,663 候选 |
| Phase 6：人工标注表 | annotation_sheet.csv | 目标先 200 行小标注，后续扩展到 1000 行；20% needs_audit | 已完成：200 行 |
| Phase 7：仲裁与评估 | adjudicated_annotations.csv, per_threshold_metrics.csv, metrics.json | joint 与 naive_union 均有 P/R/F1；能判断 Stage 4 是否优于 baseline | 未开始 |
| Phase 8：扩大规模 | 10K-50K / 100K-300K CC3M 结果 | 候选挖掘和标注达到论文主表规模 | 未开始 |
| Phase 9：LLaVA | A/B/C/D/E 训练数据与下游评测结果 | 至少 VQAv2；TextVQA 视时间加入 | 未开始 |

### 5.5 对原实验设计的当前修订说明

• 实验 1（Stage 4）目前已经完成最小可运行实现、Windows CUDA smoke 和 1K 真实 CC3M 小跑。tau=0.95 下没有直接 drop，说明自然 CC3M 1K 子集重复稀疏，后续重点应转向 candidate mining + 标注评估，而不是只看无标签 drop rate。

• 实验 2（CC3M Ground Truth）已经完成第一版候选挖掘和 200 行标注表生成。下一步需要人工填写 label/audit_label，再运行 adjudication 和 evaluation。

• 实验 3（SSCD baseline）暂未开始；当前优先级低于 Stage 4 ground truth 和 LLaVA 验证。

• 实验 4（LLaVA）暂未开始；需要等 Stage 4 真实数据去重结果和 A/B/C/D/E 数据划分准备好。

• 所有 smoke 数字只能作为工程记录，不得写入论文主表。论文主表必须来自 experiment_ledger.csv 可追踪的真实实验。

### 5.6 当前一句话结论

当前计划已经完成“工程实现 + Windows 服务器迁移验证 + 1K 真实 CC3M 数据准备 + Stage 4 小跑 + 候选挖掘 + 200 行标注表”。下一步有两条线：一是人工标注 200 行后跑 Phase 7 评估，二是把同一流程扩大到 5K/10K 为正式标注和论文实验做准备。
