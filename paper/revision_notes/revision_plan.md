# MMdedup 论文修改控制表

最后更新：2026-06-03

用途：这个文档是论文修改的控制表。后续每一处论文修改，都先在这里说明“改哪里、怎么改、为什么改、是否已经得到用户同意”，再进入 LaTeX 正文。除路径、实验 ID、原文英文句子和建议英文表述外，本文档尽量使用中文，方便快速浏览。

## 0. 当前总原则

已经确认的总方向：

- 保留原先主框架：`mixed raw data -> classification -> modality-specific cleaning for image/audio/text`。
- MMdedup 的主 claim 改为：从混合、混乱的原始数据中获取干净的 image/audio/text 单模态训练语料。
- 降低对 `MLLM` 和 `multimodal deduplication` 的过度声明。
- Stage 4 图文 pair-level dedup 作为支线亮点 / extension：展示框架能延伸到 image-text training unit，但不取代原主框架。
- 只诚实陈述系统实际解决的问题：混合数据清洗出干净单模态数据，并进一步扩展到 image-text pair-level cleaning。
- 未经用户同意，不把任何计划性修改应用到最终 LaTeX 正文。
- 当前正式修改目标只维护 IEEE 版：`paper/ieee/main.tex`。原 `paper/latex/main.tex` 暂时作为旧版参考，不作为本轮主要修改对象。
- 当前阶段不以压缩篇幅为主要目标；优先把论文口径调稳，把 Stage 4 作为清楚的支线 extension 插入正文。
- Stage 4 的写法应做到“可见但不喧宾夺主”：有方法、setup、RQ5、结果表/图、discussion 边界，但全文主线仍是从混合原始数据中获取干净的 image/audio/text corpora。

### 0.1 2026-06-03 用户确认记录

本轮用户已确认：

- A4 在写作过程中不作为强制主流程；只有当某个数字进入正文时，仍需能追溯到 source-of-truth。
- A5 改为只做好 IEEE 版，不再要求 ACM/PVLDB 版和 IEEE 版同步推进。
- G1/G2/G4/G5 确认；G3 采用降调说法。
- 章节级修改计划大体确认；Experimental Setup 可以包含 Stage 4；Evaluation Results 中 Stage 4 extension 可稍微多一些，但具体篇幅后续微调。
- Title 后续细致讨论，方向是谦虚、不过度声称。
- 图表暂不处理；如正文需要引用或提示图表修改，先留标识并在本计划中记录后续处理。
- IEEE 版直接作为正式修改版本；验证流程只编译 IEEE 版。
- 2026-06-03 进一步确认：当前不用优先考虑篇幅问题，重点是把 Stage 4 作为支线自然插入，并调整全文声明口径。

## 1. 工作假设

| ID | 假设 | 理由 | 状态 |
|---|---|---|---|
| A1 | 原 sorter + image/audio/text dedup pipeline 仍然是论文主体。 | 用户明确要求保留原始主框架和原有实验。 | 生效 |
| A2 | Stage 4 作为 extension，而不是新中心。 | 避免论文看起来像推翻原方案后重写。 | 生效 |
| A3 | MLLM 保留为应用背景，不作为最强 claim。 | 当前主体主要是清洗数据流，不是完整 MLLM 多模态 dedup 系统。 | 生效 |
| A4 | Stage 4 结果若写入论文，仍建议能追溯到 source-of-truth 文件。 | 写作过程中不作为强制主流程；但正式数字进入正文时仍要避免无来源数字。 | 弱化生效 |
| A5 | 本轮只维护 IEEE 版。 | 用户明确要求只做好 IEEE 这一版。 | 生效 |

## 2. 全局 claim 调整

| ID | 当前 claim 类型 | 建议改法 | 修改原因 | 状态 |
|---|---|---|---|---|
| G1 | `"MMdedup is a multimodal data deduplication framework for MLLM training."` | `"MMdedup is an end-to-end data cleaning framework that classifies heterogeneous raw data and cleans image/audio/text corpora; we further extend it to image-text pair-level deduplication."` | 更诚实：保留原贡献，同时把 Stage 4 放在 extension 位置。 | 已确认 |
| G2 | `"We solve multimodal deduplication."` | `"We solve clean modality-specific corpus acquisition from mixed raw data, and demonstrate an extension to image-text pair cleaning."` | 避免暗示每个模块都做 cross-modal reasoning。 | 已确认 |
| G3 | `"First multimodal deduplication framework for MLLM training."` | 删除或降调为 `"we present a practical framework for mixed-modality data cleaning"`。 | `first` 和泛化 MLLM claim 都容易被 reviewer 攻击。 | 已确认：采用降调说法 |
| G4 | Stage 4 被写成 MMdedup 的主要证明。 | Stage 4 只作为框架可扩展性和 image-text 处理能力的补充证据。 | 匹配用户要求：主框架第一，Stage 4 第二。 | 已确认 |
| G5 | VQA/COCO downstream 被写成决定性 MLLM 证据。 | downstream caption/VQA 只作为辅助验证；Stage 4 的直接证据仍是 pair-level detection。 | 避免过度解释 downstream transfer。 | 已确认 |

## 3. 章节级修改计划

| 章节 | 当前作用 | 修改后作用 | 主要动作 | 状态 |
|---|---|---|---|---|
| Title | 强调 MLLM training 和 multimodal dedup framework。 | 强调 mixed-modality data cleaning；MLLM 可不放标题。 | 后续单独讨论，方向是谦虚标题。 | 后续细化 |
| Abstract | 把 MMdedup 说成 MLLM multimodal dedup framework。 | 主写混合原始数据清洗；Stage 4 简短出现，并可加入 Stage 4 表。 | 重写。 | 已确认 |
| Introduction | 从 MLLM 和 multimodal dedup gap 切入。 | 从 heterogeneous web data / clean corpus construction 切入，MLLM 作为背景。 | 改 problem statement、gap、system paragraph、contributions。 | 已确认 |
| Contributions | 有较强 `first multimodal` 倾向。 | 改成：框架、Sorter、三类 dedup 模块、Stage 4 extension。 | 重写 bullet。 | 已确认 |
| Related Work | 对比单模态 dedup 和 file type detection。 | 保留主体；补 Stage 4 需要的 image-text curation / pair cleaning 相关定位。 | 中等幅度修改。 | 已确认 |
| System Design | 只有两阶段 classification-and-clean。 | 保留两阶段核心；在 Phase 2 后加 Stage 4 extension 小节，稍微说明图文处理延伸。 | 增加但不喧宾夺主。 | 已确认 |
| Experimental Setup | 只覆盖 sorter 和三类单模态 dedup。 | 保留原实验；补 Stage 4 dataset/baselines/metrics。 | 可比原计划稍多一些。 | 已确认 |
| Evaluation Results | Sorter、dedup effectiveness、threshold、downstream、ablation。 | 原实验仍为主；加 Stage 4 extension 结果小节。 | Stage 4 可以稍微多一点，具体篇幅后续微调。 | 已确认 |
| Discussion | 把 cross-modal duplicates 作为未来工作。 | 改为：核心 pipeline 是单模态清洗，Stage 4 已覆盖 image-text，其他 cross-modal 仍是未来工作。 | 小幅重写。 | 已确认 |
| Conclusion | 继续强调 MLLM/multimodal cleaning。 | 总结 clean data acquisition framework + image-text extension，并加入 Stage 4 extension 发现。 | 轻量重写。 | 已确认 |

## 4. 逐项修改清单

说明：`Applied` 必须保持 `No`，直到用户明确同意对应修改后才能改 LaTeX。

| ID | 位置 | 当前段落作用 | 建议怎么改 | 修改原因 | 可靠性 | 审批 | Applied |
|---|---|---|---|---|---|---|---|
| P-TITLE-01 | `paper/ieee/main.tex`, title | 把论文定位为 MLLM-focused multimodal dedup framework。 | 题目可改向 mixed-modality data cleaning，例如 `"MMdedup: An End-to-End Framework for Sorting and Deduplicating Mixed-Modality Data"`；最终标题后续单独讨论，方向是谦虚、不过度声称。 | 现标题过度绑定 MLLM 和 multimodal dedup。 | 中：需要后续决定题目风格。 | 后续细化 | No |
| P-ABS-01 | `paper/ieee/main.tex`, abstract opening | 从 MLLM training data cleaning 开始。 | 从 raw heterogeneous web data 和 clean corpus acquisition 开始；MLLM 只作为 downstream context。 | 降低过度声明，问题更稳。 | 高：建议可靠。 | 方向已确认 | No |
| P-ABS-02 | `paper/ieee/main.tex`, abstract system claim | 定义为 `"a highly efficient multimodal data deduplication framework"`。 | 改成 classification-and-clean framework，把 mixed raw files 清洗成 image/audio/text corpora。 | 区分 mixed-modality handling 和 cross-modal reasoning。 | 高。 | 方向已确认 | No |
| P-ABS-03 | `paper/ieee/main.tex`, abstract evidence | 只列 sorter + 三类 dedup 结果。 | 保留原主实验数字；加入 Stage 4 表后，在摘要用一句话提 Stage 4 extension。 | Stage 4 要出现，但不能压过主线。 | 高。 | 方向已确认：加入 Stage 4 表 | No |
| P-ABS-04 | `paper/ieee/main.tex`, abstract final sentence | 用 near-duplicate detection 证明 core design。 | 改为 experiments validate classification-and-clean design；Stage 4 只证明 image-text extension。 | 避免读者误以为所有模块都是 joint multimodal。 | 高。 | 方向已确认 | No |
| P-INTRO-01 | `paper/ieee/main.tex`, introduction opening | 以 LLM/MLLM 数据质量作为开头。 | 保留数据质量动机，但主语改为 heterogeneous web data 和 clean data acquisition；MLLM 后移为应用背景。 | 匹配“获取干净数据”的新主线。 | 高。 | 方向已确认 | No |
| P-INTRO-02 | `paper/ieee/main.tex`, digital swamp paragraph | 提出 heterogeneity 和 redundancy 两个挑战。 | 保留两挑战结构，只收紧为 mixed raw files 和 clean modality-specific corpora。 | 这是原文最强的主线，应保留。 | 高。 | 方向已确认 | No |
| P-INTRO-03 | `paper/ieee/main.tex`, gap | 说现有 NDD 只处理 pre-classified 单模态，缺少 integrated front end。 | 保留这个 gap，但称为 mixed-modality data cleaning gap，不说完整 multimodal dedup gap。 | 避免 reviewer 说“你只是三个单模态 pipeline”。 | 高。 | 方向已确认 | No |
| P-INTRO-04 | `paper/ieee/main.tex`, system paragraph | 介绍 MMdedup 为 MLLM training 的 multimodal dedup framework。 | 改成：先 classify heterogeneous raw inputs，再做 modality-aware exact/near dedup，输出 clean image/audio/text corpora；当 training unit 是 paired sample 时，可接 image-text pair extension。 | 建立主框架和 Stage 4 的层级关系。 | 高。 | 方向已确认 | No |
| P-INTRO-05 | `paper/ieee/main.tex`, novelty sentence | 说 novelty 在 individual components 和 MLLM cleaning integration。 | 改为 `"end-to-end orchestration, robust routing of mixed raw files, empirical validation of modality-specific cleaning, and an extensible path to image-text pair cleaning"`。 | 更准确，不夸大。 | 高。 | 方向已确认 | No |
| P-INTRO-06 | `paper/ieee/main.tex`, first claim | `"first research work on multimodal data deduplication for MLLM training"`。 | 删除 `first`，改为 practical framework / end-to-end framework。 | `first` 高风险且无必要。 | 高：强烈建议。 | 方向已确认 | No |
| P-CONTRIB-01 | `paper/ieee/main.tex`, contribution 1 | 声称 parallel/pipelined multimodal dedup framework。 | 改成 mixed-modality classification-and-clean framework，用于从 raw heterogeneous data 获得 clean image/audio/text corpora。 | 主贡献要干净明确。 | 高。 | 方向已确认 | No |
| P-CONTRIB-02 | `paper/ieee/main.tex`, sorter bullet | Sorter 贡献。 | 保留，强调 content-aware routing、误导 extension、JSON wrapper、高 throughput。 | 实验支撑充分。 | 高。 | 方向已确认 | No |
| P-CONTRIB-03 | `paper/ieee/main.tex`, module bullet | audio dedup + modality-aware technique 混在一起。 | 拆清楚或改成 integrate scalable near-dedup modules for image/audio/text；避免暗示每个算法都是新算法。 | 比 `"rather than proposing novel algorithms"` 更正向、更诚实。 | 高。 | 方向已确认 | No |
| P-CONTRIB-04 | `paper/ieee/main.tex`, new bullet | 当前 active text 没有 Stage 4 贡献。 | 增加一条 Stage 4 image-text pair-level cleaning extension，引用 source-traceable evaluation，但明确为 extension evidence。 | 让 Stage 4 可见但不喧宾夺主。 | 中高：具体篇幅后续微调。 | 方向已确认 | No |
| P-RELATED-01 | `paper/ieee/main.tex`, related work opening | 从 MLLM training data selection 切入。 | 改为 large-scale ML/AI training corpora from heterogeneous web crawls；MLLM 作为例子。 | 和主 claim 对齐。 | 高。 | 方向已确认 | No |
| P-RELATED-02 | `paper/ieee/main.tex`, related-work table | 比较 data cleaning、dedup、file type detection、MMdedup。 | 保留表，但检查 `"Our Paper"` 是否改为 `"MMdedup"`；确认 Mixed/Auto/Raw/E2E/Config columns 是否支持新 claim。 | 表可以支撑 mixed-data framework。 | 中：需要看最终表格版面。 | 方向已确认 | No |
| P-RELATED-03 | `paper/ieee/main.tex`, text/image dedup | 综述单模态 text/image dedup。 | 基本保留；过渡句强调这些是 MMdedup 集成的组件，不是完全取代前人算法。 | 让 integration 贡献可信。 | 高。 | 方向已确认 | No |
| P-RELATED-04 | `paper/ieee/main.tex`, audio dedup | 说 audio 在 ML training data curation 中较少被处理。 | 保留，但避免暗示解决所有 multimodal audio 问题；说它填补 mixed-data framework 的 audio-cleaning branch。 | 保留 audio 贡献但不扩张。 | 高。 | 方向已确认 | No |
| P-RELATED-05 | `paper/ieee/main.tex`, image-text related work | 相关内容目前在 comment 中，不编译。 | 最终会写一个短 Stage 4 相关定位，讲 LAION/DataComp/image-text curation，并区分 filtering 和 pair-level duplicate detection。 | 给 Stage 4 合理 related work 背景。 | 中高：篇幅后续控制。 | 方向已确认：最终写 Stage 4 | No |
| P-RELATED-06 | `paper/ieee/main.tex`, positioning | `"Rather than proposing novel algorithms..."`。 | 删除这类削弱性表达，改成正向定位：MMdedup contributes practical orchestration and validated modality-specific cleaning pipeline。 | 当前句子主动削弱 novelty。 | 高：强烈建议。 | 方向已确认 | No |
| P-RELATED-07 | `paper/ieee/main.tex`, final phrase | `"preparing multimodal training data at scale"`。 | 改为 preparing clean modality-specific corpora from mixed raw data, with an extension to paired image-text units。 | 和新边界一致。 | 高。 | 方向已确认 | No |
| P-SYSTEM-01 | `paper/ieee/main.tex`, System Design opening | 定义两阶段框架。 | 保留核心架构，明确 core output 是 clean modality-specific corpora，并稍微说明框架有图文 pair-level 处理延伸。 | 原框架不变，同时回应 Stage 4 extension。 | 高。 | 方向已确认 | No |
| P-SYSTEM-02 | `paper/ieee/main.tex`, Design Rationale | 说 deep cross-modal alignment 成本高，不适合初始清洗。 | 保留 lightweight-first，但改成“深层 pair-level cleaning 留给 refined paired data 的 extension”，避免和 Stage 4 矛盾。 | Stage 4 加入后，原句需要协调。 | 高。 | 方向已确认 | No |
| P-SYSTEM-03 | `paper/ieee/main.tex`, Sorter | Sorter 内部用 Stage 1/Stage 2。 | 可把 sorter 内部的 Stage 改为 Step/Layer，避免和 Stage 4 extension 撞名。 | 减少术语混乱。 | 中高：建议但不强制。 | 方向已确认 | No |
| P-SYSTEM-04 | `paper/ieee/main.tex`, Phase 2 modules | 三类单模态 near-dedup。 | 保留主体，在结尾加桥接句：clean modality-specific streams 之后，可对 paired image-caption data 应用 Stage 4 extension。 | 平滑引出 Stage 4。 | 高。 | 方向已确认 | No |
| P-SYSTEM-05 | Phase 2 后新增 subsection | active System Design 中没有 Stage 4。 | 新增 `"Extension: Image-Text Pair-Level Deduplication"`，简述 pair id、image/text embeddings、joint score 或 conservative rule、keep/drop outputs、quality-aware retention。 | 在正确层级展示新能力。 | 高：具体长度后续微调。 | 方向已确认 | No |
| P-SYSTEM-06 | `tab:notation` | 只包含 text/image/audio。 | 建议不塞进主 notation table；另用简短 Stage 4 notation paragraph 或小表。 | 避免主表膨胀。 | 中：取决于版面。 | 方向已确认 | No |
| P-SYSTEM-07 | Algorithm placement | 当前只有三个单模态 algorithm。 | Stage 4 推荐 prose + formula/table，不建议大 algorithm，除非页面允许。 | Stage 4 是 side highlight，大算法会让它显得过重。 | 中：版面驱动。 | 方向已确认 | No |
| P-SETUP-01 | `paper/ieee/main.tex`, RQs | 只有 RQ1-RQ4。 | 加一个 secondary RQ5：`Can the image-text extension improve pair-level duplicate detection over modality-wise baselines?` | 实验问题要覆盖新增 Stage 4 结果。 | 高。 | 方向已确认：写结果 | No |
| P-SETUP-02 | `paper/ieee/main.tex`, Datasets | 只有五个原数据集。 | 加 CC3M-derived image-text pair evaluation：3000 score-space stratified labeled pair-pairs；不要把旧 1000 high-joint 写成正文 setup。 | Stage 4 表出现就必须交代数据，同时避免旧诊断集混入主证据。 | 高。 | 方向已确认：不要旧 1000 | No |
| P-SETUP-03 | `paper/ieee/main.tex`, Baselines | 只有 file/image/audio/text baselines。 | 增加 Stage 4 baselines：image-only、text-only、naive union、joint/pair-level 或 conservative pair rule。 | 和 Stage 4 指标对应。 | 高。 | 方向已确认 | No |
| P-SETUP-04 | `paper/ieee/main.tex`, Metrics | 指标覆盖 classification/dedup/efficiency/downstream。 | 加 pair-level precision/recall/F1；caption metrics 只作为 auxiliary。 | Stage 4 直接证据需要这些指标。 | 高。 | 方向已确认 | No |
| P-SETUP-05 | `paper/ieee/main.tex`, Implementation Details | 原模块硬件和超参。 | 加 Stage 4 简短实现说明：CLIP image/text features、fixed thresholds、score-space stratified annotation、experiment IDs。 | 可复现但不占太多篇幅。 | 中高。 | 方向已确认 | No |
| P-RESULTS-01 | `paper/ieee/main.tex`, Results overview | 结果回答 RQ1-RQ4。 | 加入 RQ5，并在结果 overview 中同步提到 Stage 4 extension；RQ5 的含义见下方专门说明。 | 结构一致性。 | 高。 | 方向已确认：独立加入 RQ5 | No |
| P-RESULTS-02 | `paper/ieee/main.tex`, original results | 原实验主结果。 | 保留为主，只把误用的 `multimodal` 改成 `mixed-modality` 或 `modality-specific`。 | 保护原论文主体。 | 高。 | 方向已确认 | No |
| P-RESULTS-03 | 新 Stage 4 result subsection | 当前 active paper 没有 Stage 4 结果。 | 在 ablation 后或 discussion 前加 Stage 4 小节：3000 fair labels，image/text/naive-union vs Stage 4 pair rule，P/R/F1，可选 CI；具体篇幅后续微调。 | 给 Stage 4 可见证据。 | 高。 | 方向已确认：篇幅后续微调 | No |
| P-RESULTS-04 | Stage 4 detection numbers | 当前论文没有。 | 可用源文件数字：image F1 0.333，text F1 0.279，naive union F1 0.322，joint F1 0.616，conservative_and F1 0.541；需要决定 joint 作为 detection result，conservative_and 作为 training operating point。 | 避免混淆两个 operating point。 | 高：数字已从本地 CSV 复核。 | 方向已确认 | No |
| P-RESULTS-05 | COCO Caption downstream | 当前论文没有。 | 如加入，只写辅助验证：E_train_stage4_conservative 保留 190,744 pairs，COCO near-best；D_naive_union CIDEr/ROUGE-L 略高但只保留 155,127 pairs；E BLEU-4 更高。 | 诚实表达 tradeoff。 | 高：数字已从本地 CSV 复核。 | 方向已确认 | No |
| P-RESULTS-06 | VQAv2/random results | 当前 active paper 没有，但 Stage 4 artifacts 中有。 | 不单独作为旁支结果；若需要相关内容，放入主线 discussion，用来说明 downstream diagnostic 的边界。 | 避免把不稳定诊断拖进主叙事。 | 高。 | 方向已确认：放入主线 discuss | No |
| P-RESULTS-07 | source-of-truth IDs | 当前正文不写 experiment IDs。 | drafting 阶段可在 table caption/notes 加 experiment ID；camera-ready 是否保留另议。 | 便于追踪数字。 | 中。 | 方向已确认 | No |
| P-DISC-01 | `paper/ieee/main.tex`, Efficiency | 讨论可扩展性，并说适合 MLLM training data preparation。 | 泛化成 large-scale training-data preparation；只有 Stage 4 段落才提 paired image-text。 | 降低 MLLM 过度绑定。 | 高。 | 方向已确认 | No |
| P-DISC-02 | `paper/ieee/main.tex`, Limitations | 说 pipeline treats modalities independently and may miss cross-modal duplicates。 | 改成 core pipeline by design 是 modality-specific；Stage 4 image-text extension 覆盖一个 paired-data case，其他 cross-modal/audio-text/streaming 仍是未来工作。 | 与 Stage 4 加入后保持一致。 | 高。 | 方向已确认 | No |
| P-DISC-03 | `paper/ieee/main.tex`, Future Work | 提 multimodal embeddings for cross-modal duplicate detection。 | 收窄为 broader pair types、better keeper quality scoring、automatic threshold selection、incremental/streaming index。 | 更具体，不泛泛而谈。 | 高。 | 方向已确认 | No |
| P-CONCL-01 | `paper/ieee/main.tex`, conclusion first paragraph | 复述 multimodal dedup for MLLM training。 | 改成 mixed-modality data cleaning framework 和 clean image/audio/text corpora；加入 Stage 4 extension。 | 结论必须和新主线一致。 | 高。 | 方向已确认：加入 Stage 4 | No |
| P-CONCL-02 | `paper/ieee/main.tex`, conclusion evidence | 列原实验 key findings。 | 保留 sorter 和三模块发现；加一句 Stage 4 extension 的发现。 | 证据层级清楚。 | 高。 | 方向已确认：加入 Stage 4 extension 发现 | No |
| P-COMMENT-01 | `paper/ieee/main.tex`, commented old drafts | 大量旧稿在 comment 中。 | 暂不删；所有内容修改获批后再决定是否清理。 | 防止无关 diff 变大。 | 高：建议 defer。 | 后续决定 | No |
| P-FIG-01 | `figures/figure00.pdf` | taxonomy figure 可能暗示 multiple modality/data swamp。 | 图表方面暂不处理；若正文需要引用或提示后续改图，先留标识并记录。 | 图可能和降调 claim 冲突，但当前不是优先项。 | 中：后续视觉审阅。 | 暂缓 | No |
| P-FIG-02 | `figures/figure1.pdf` | architecture figure 是原两阶段框架。 | 图表方面暂不处理；Stage 4 先用文字说明，若需要 extension box 再记录后续处理。 | 不轻易大改图。 | 中：后续看图源。 | 暂缓 | No |
| P-IEEE-01 | `paper/ieee/main.tex` | IEEE 副本目前是旧内容的格式转换。 | 直接在 IEEE 版上改动。 | 用户明确要求只做好 IEEE 这一版。 | 高。 | 方向已确认 | No |
| P-VERIFY-01 | `paper/ieee/main.pdf` | 还没有应用内容修改。 | 每轮获批修改后，只编译 `paper/ieee/main.tex` 并检查 PDF 渲染。 | 本轮只维护 IEEE 版。 | 高。 | 方向已确认 | No |

### 4.1 RQ5 的含义

正文确认加入 RQ5。它不是要把 Stage 4 变成全文中心，而是给 Stage 4 extension 一个清楚、可回答的实验问题。

建议 RQ5：

> RQ5: Can the image-text extension improve pair-level duplicate detection over modality-wise baselines?

中文含义：

- 它问的是：当数据单元从单个 image/audio/text 文件变成 image-caption pair 时，Stage 4 这个图文 pair-level extension 是否比只看 image、只看 text、或者把 image-only/text-only drops 做 naive union 更有效。
- 它的直接证据是 3000 fair score-space stratified labeled pair-pairs 上的 precision/recall/F1。
- 它不负责证明 MMdedup 是完整 MLLM dedup 系统，也不负责证明 Stage 4 在所有 downstream task 上最好。
- 它在文章结构里的角色是 secondary RQ / extension RQ：主线 RQ1-RQ4 仍然服务于 sorter 和 image/audio/text cleaning，RQ5 只服务于 Stage 4 extension。

备选方案已放弃：不采用不编号的 extension question，正文使用独立 RQ5。

> Extension question: Does pair-level image-text cleaning improve duplicate detection beyond modality-wise composition?

## 5. 术语替换规则

本节替换方向已全部确认。后续写具体英文段落时按这些规则执行。

| 当前常见说法 | 建议替换 | 使用位置 | 原因 |
|---|---|---|---|
| `"multimodal data deduplication framework"` | `"mixed-modality data cleaning framework"` 或 `"classification-and-clean framework"` | MMdedup 主 claim。 | 避免暗示全系统都做 cross-modal reasoning。 |
| `"for MLLM training"` | `"for large-scale training-data preparation"` 或 `"for ML/AI training corpora"` | 标题、摘要、结论、泛化系统表述。 | MLLM 是背景，不是所有实验的直接对象。 |
| `"MLLM"` | `"ML/AI training"` 或 `"large-scale model training"` | 通用动机和实验框架。 | 不让每个结果都变成 MLLM-specific。 |
| `"multimodal web harvests"` | `"heterogeneous raw web data"` 或 `"mixed-modality web data"` | 描述输入。 | 保留 digital swamp 主线，同时降调。 |
| `"across modalities"` | `"across mixed input streams"` 或 `"for image/audio/text streams"` | 核心 pipeline。 | 明确核心模块是在分类后的 stream 上处理。 |
| `"first research work"` | 删除，或改成 `"we present a practical framework"` | Introduction/contributions。 | `first` 风险高。 |
| `"rather than proposing novel algorithms"` | 删除，改成正向贡献表达。 | Related Work positioning。 | 这句话主动削弱 novelty。 |
| `"cross-modal duplicate detection"` | `"image-text pair-level extension"` | 仅 Stage 4。 | Stage 4 只覆盖实现过的 image-text case。 |
| `"best downstream performance"` | `"near-best / preserves performance under a retention tradeoff"` | COCO/VQA downstream。 | D/E 在不同指标上有 tradeoff。 |

## 6. 建议的新论文叙事

本节叙事方向已确认。

建议总 thesis：

> MMdedup is an end-to-end classification-and-clean framework that converts heterogeneous raw web data into clean image, audio, and text corpora. Its core contribution is robust mixed-input routing plus validated modality-specific exact and near-duplicate removal. As an extension, MMdedup can also operate on image-text training units through a Stage 4 pair-level cleaning module.

建议层级：

1. 主问题：raw heterogeneous data 很难直接变成干净训练语料。
2. 主框架：先 classify mixed files，再 clean image/audio/text streams。
3. 主证据：sorter、三类 dedup quality、threshold behavior、downstream utility、ablation。
4. 支线亮点：Stage 4 image-text pair-level extension 证明框架可延伸到 paired multimodal training data。
5. 局限：核心 pipeline 不是 universal cross-modal dedup solution；Stage 4 只覆盖 image-text pairs。

## 7. 建议的 active paper 结构

| 顺序 | 章节 | 动作 | 备注 |
|---|---|---|---|
| 1 | Title | 讨论是否降调。 | 候选：`"MMdedup: An End-to-End Framework for Sorting and Deduplicating Mixed-Modality Data"`。 |
| 2 | Abstract | 重写。 | 主框架先写，Stage 4 一句话；可加入 Stage 4 表的核心发现。 |
| 3 | Introduction | 改 opening/gap/system/contribution。 | 保留 digital swamp 叙事。 |
| 4 | Related Work | 中等修改。 | 补 Stage 4 的 image-text curation 相关定位。 |
| 5 | System Design | 保留 Phase 1/2，加 Stage 4 extension subsection。 | 稍微说明图文处理延伸，但 Stage 4 不应看起来像整个系统。 |
| 6 | Experimental Setup | 保留原实验设置，补 Stage 4 dataset/baselines/metrics。 | 可以比原计划稍多一些，但不写旧 1000 high-joint 为正文 setup。 |
| 7 | Evaluation Results | 保留原结果，加 Stage 4 extension result。 | Stage 4 可以稍微多一点；具体篇幅后续微调。 |
| 8 | Discussion | 更新 limitations/future work。 | 承认核心设计是 modality-specific。 |
| 9 | Conclusion | 轻量重写。 | 加入 Stage 4 extension 发现，并和证据一致。 |

## 8. Stage 4 证据使用边界

Stage 4 只作为 extension 使用。A4 已弱化为写作流程中的非强制项；但任何进入正文的具体数字，仍应尽量从本地 source-of-truth 文件核对。

| 证据 | 来源 | 建议用途 | 注意事项 |
|---|---|---|---|
| 3000 fair score-space stratified labels completed | `docs/stage4_dashboard/data/latest_annotation_status.json`; `experiments/results/plan_b_stage4/exp_stage4_fair_annotation_3000_20260523/` | Dataset construction / annotation protocol。 | 正文 setup 写 3000 fair set；旧 1000 high-joint 不写入正文 setup。 |
| Fixed-threshold 3000-label Stage 4 evaluation | `experiments/results/plan_b_stage4/exp_stage4_fair_eval_3000_conservative_and_20260601/fixed_threshold_metrics_with_conservative_and.csv` | Stage 4 extension 主表。 | `joint` 和 `conservative_and` 不能混成一个规则。 |
| Bootstrap CI over 3000 fair labels | `experiments/results/plan_b_stage4/exp_stage4_fair_eval_3000_bootstrap_ci_20260531/metrics.json` | 可选 CI 句子或表注。 | CI 支持这个 stratified set，不代表真实分布 prevalence。 |
| COCO Caption A/B/C/D/E | `experiments/results/plan_b_stage4/icdm_revision/summary_20260530/llava_coco_caption_val2014_5k_ckpt1500_20260601.csv` | 只能作为辅助 downstream/transfer validation。 | 不能说 Stage 4 best on all downstream metrics。 |
| Stage 4 efficiency summary | `experiments/results/plan_b_stage4/exp_stage4_efficiency_summary_20260530/` | 若篇幅允许，可进 discussion/table。 | 是 derived summary，最终数字要查 per-row source metrics。 |

推荐写法：

- 可写：`"The image-text extension improves pair-level duplicate detection on a 3000-row score-space stratified evaluation set."`
- 可写：`"The conservative Stage 4 training split preserves many more pairs than the naive-union split while obtaining near-best COCO Caption transfer."`
- 避免：`"Stage 4 is best on all downstream tasks."`
- 避免：`"MMdedup fully solves multimodal deduplication for MLLM training."`
- 避免：`"VQAv2/random-25K proves or disproves Stage 4."`

## 9. Stage 4 operating point 规则

不要把两个 Stage 4 规则混成一句话。

| 规则 | 指标用途 | 最适合写在哪里 | 当前源文件数值 |
|---|---|---|---|
| `joint >= 0.85` | pair-level duplicate/near-duplicate detection。 | Stage 4 direct detection result。 | Precision 0.569，recall 0.671，F1 0.616。 |
| `image >= 0.85 AND text >= 0.85` | conservative training-data cleaning。 | training split / downstream transfer tradeoff。 | Precision 0.726，recall 0.431，F1 0.541。 |
| `image >= 0.85 OR text >= 0.95` | naive union baseline。 | independent modality composition baseline。 | F1 0.322。 |

推荐写法：

> For pair-level detection, the joint-score rule provides the strongest F1 on the 3000-label fair evaluation set. For downstream training-data construction, we also evaluate a conservative Stage 4 operating point that favors precision and data retention.

## 10. COCO Caption 写法规则

如果最终论文写 COCO Caption，只作为辅助证据。

源文件：

`experiments/results/plan_b_stage4/icdm_revision/summary_20260530/llava_coco_caption_val2014_5k_ckpt1500_20260601.csv`

| Split | Kept Pairs | Dedup Rate | CIDEr | BLEU-4 | ROUGE-L | 建议解释 |
|---|---:|---:|---:|---:|---:|---|
| A raw | 200000 | 0.0000 | 0.686 | 0.164 | 0.428 | 原始参考。 |
| B image-only | 177796 | 0.1110 | 0.683 | 0.160 | 0.415 | 轻度清洗，caption transfer 接近 raw。 |
| C text-only | 170783 | 0.1461 | 0.674 | 0.162 | 0.417 | text-only baseline。 |
| D naive-union | 155127 | 0.2244 | 0.749 | 0.184 | 0.443 | 更激进的 cleaning baseline，CIDEr/ROUGE-L 略高。 |
| E Stage4 conservative | 190744 | 0.0463 | 0.742 | 0.190 | 0.439 | 保留更多 pairs，同时 caption transfer near-best，BLEU-4 更高。 |

推荐写法：

> On COCO Caption, the conservative Stage 4 split remains close to the strongest caption-transfer baseline while preserving substantially more training pairs. The result supports Stage 4 as a conservative cleaning extension rather than a universal downstream-performance winner.

## 11. 暂时不要做的事

| 候选动作 | 建议 | 原因 |
|---|---|---|
| 把 Stage 4 改成全文中心。 | 不做。 | 用户要求保留原主框架。 |
| 删除 audio。 | 不做。 | audio 是原三模态框架的一部分。 |
| 用 Stage 4 实验替换所有原实验。 | 不做。 | 主证据仍是 sorter + 三类 cleaning modules。 |
| 声称 `"first multimodal dedup framework"`。 | 不做。 | 高风险且无必要。 |
| 主推 random-25K 或 `BASE_no_adapter`。 | 不做。 | 它们是诊断结果，容易扰乱主线。 |
| 在 claim 对齐前大规模润色全文。 | 不做。 | 先定叙事，再改语言。 |
| 立即清理大块 inactive `comment`。 | 暂缓。 | 和 claim 修正无关，会制造 noisy diff。 |

## 12. 可靠性复审

### 12.1 我认为可以直接推进的修改点

这些修改点和用户方向一致，并且不依赖新增实验：

- P-ABS-01 至 P-ABS-04：摘要降调，改成 mixed raw data cleaning 主线。
- P-INTRO-01 至 P-INTRO-06：引言中移除过强 MLLM / first / multimodal dedup claim。
- P-CONTRIB-01 至 P-CONTRIB-03：贡献列表重写为框架、Sorter、三类 cleaning modules。
- P-RELATED-06：删除 `"Rather than proposing novel algorithms..."` 这种削弱表达。
- P-DISC-01 至 P-DISC-03：讨论部分承认 core pipeline 是 modality-specific，并把 Stage 4 放成 image-text extension。
- P-CONCL-01 至 P-CONCL-02：结论跟随新主线重写。

可靠性判断：高。原因是这些是 claim alignment，不需要额外数据，只需要把现有叙事和实际系统对齐。

### 12.2 已确认但需要后续细化的修改点

这些点方向已经确认，但具体英文写法、篇幅或标题风格还需要后续逐段定稿：

- P-TITLE-01：标题后续单独讨论；方向是谦虚、不过度声称。
- P-CONTRIB-04：Stage 4 可以作为 extension contribution 出现；具体一句还是一个 bullet 后续看篇幅。
- P-RELATED-05：最终会写 Stage 4 相关定位；篇幅需控制。
- P-SYSTEM-05 至 P-SYSTEM-07：Stage 4 方法会写；具体是 prose、formula、小表还是 algorithm 后续看版面。
- P-RESULTS-03 至 P-RESULTS-05：Stage 4 和 COCO Caption 进入结果部分；Stage 4 篇幅可稍多，具体篇幅后续微调。

可靠性判断：中高。原因是内容本身有证据，但篇幅和叙事权重需要你定。

### 12.3 暂缓处理的图表项

- P-FIG-01：暂不处理 taxonomy 图；如正文需要引用或提示后续改图，先留标识并记录。
- P-FIG-02：暂不处理 architecture 图；Stage 4 先用文字说明，必要时后续再加 extension box。

可靠性判断：中。原因是这些不是逻辑问题，而是版面和素材问题。

### 12.4 我重新审阅后认为需要保守处理的点

- Stage 4 的 `joint >= 0.85` 和 `conservative_and` 不应混用。前者更适合写 pair-level detection F1，后者更适合写 training-data cleaning / downstream transfer tradeoff。
- COCO Caption 结果不能写成 Stage 4 全面胜出。D 在 CIDEr/ROUGE-L 上略高，E 在 BLEU-4 上更高且保留更多数据。
- VQAv2/random-25K 不应进入主线。它们容易把论文从“干净数据获取框架”带偏到不稳定 downstream 诊断。
- 原始三模态实验虽然有些表述需要降调，但不要删掉；它们是论文主体证据。

## 13. 后续编辑流程

每一轮正式改论文时按这个顺序：

1. 选定一个 section，例如 Abstract。
2. 在本控制表中补充该 section 的具体英文改写草稿。
3. 用户确认。
4. 修改 `paper/ieee/main.tex`。
5. 编译 IEEE PDF：
   - `cd paper/ieee && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
6. 检查 IEEE PDF 渲染。
7. 回填本表：`Applied = Yes`，并写 verification 结果。

## 14. 需要维护的输出版本

| 版本 | 路径 | 用途 | 备注 |
|---|---|---|---|
| IEEE source | `paper/ieee/main.tex` | 本轮正式修改源文件。 | 后续正文修改直接在这里做。 |
| IEEE PDF | `paper/ieee/main.pdf` | 本轮正式预览 PDF。 | 每轮正式修改后重编译并检查渲染。 |
| 旧版参考 source | `paper/latex/main.tex` | 旧格式参考。 | 本轮不主动维护。 |
| 旧版参考 PDF | `paper/latex/main.pdf` | 旧格式参考渲染。 | 本轮不主动重编译。 |

## 15. 2026-06-03 第一轮正文应用记录

本轮实际修改文件：

- `paper/ieee/main.tex`
- `paper/ieee/main.pdf`

本轮没有修改：

- `paper/latex/main.tex`
- `paper/latex/main.pdf`
- `paper/ieee/references.bib`

引用处理原则：

- 没有删除现有 citation key。
- 新增的 image-text related work 最终只保留 active citation key：`schuhmann2022laion`、`gadre2023datacomp`。第一版曾启用 `fang2023data`、`evans2024data`，但已在第 16 节记录中按用户确认撤回。
- 没有新增、删除或替换 BibTeX 条目。

| 对应 ID | 实际怎么改 | 修改原因 | Applied |
|---|---|---|---|
| P-ABS-01 至 P-ABS-04 | 重写 abstract：从 raw heterogeneous web data / clean modality-specific corpora 开始，MLLM 不再作为主 claim；加入 Stage 4 image-text extension 和 3,000-label F1 结果。 | 摘要需要立刻呈现新主线，并让 Stage 4 可见但不压过主框架。 | Yes |
| 术语替换规则 | Keywords 改为 `mixed-modality data cleaning, data deduplication, near-duplicate detection, image-text pair deduplication`。 | 降低 `multimodal data deduplication for MLLM training` 的过度声明。 | Yes |
| P-INTRO-01 至 P-INTRO-06 | 重写 introduction 前三段：从 large-scale training data quality 和 raw web digital swamp 切入；把 `multimodal web harvests` 改成 `mixed-modality web harvests`；删除 `first research work`。 | 主问题改成 clean data acquisition，而不是泛化 MLLM 多模态去重。 | Yes |
| P-CONTRIB-01 至 P-CONTRIB-04 | 重写 contribution bullets：框架、Sorter、三类 cleaning modules、Stage 4 extension、prototype/evaluation。 | 贡献层级变为主框架 + 支线 extension。 | Yes |
| P-RELATED-01、P-RELATED-05、P-RELATED-06、P-RELATED-07 | Related Work opening 降调；新增 active `Image-Text Dataset Curation` 小节；删除削弱性表达 `"Rather than proposing novel algorithms..."`。 | 给 Stage 4 合理相关工作背景，同时避免自我削弱 novelty。 | Yes |
| P-SYSTEM-01、P-SYSTEM-02、P-SYSTEM-05、P-SYSTEM-07 | System Design opening 和 rationale 加入 optional Stage 4；新增 `Extension: Image-Text Pair-Level Deduplication` prose 小节。 | 保持两阶段 core pipeline，同时解释图文 pair-level extension 的位置。 | Yes |
| P-SETUP-01 至 P-SETUP-05 | Experimental Setup 加入 RQ5、CC3M-PairEval、Stage 4 baselines、pair-level metrics、COCO Caption 辅助指标、Stage 4 threshold 说明。 | Stage 4 结果进入正文后，setup 必须交代数据、baseline、metric 和实现边界。 | Yes |
| P-RESULTS-01、P-RESULTS-03 至 P-RESULTS-06 | Evaluation Results overview 加入 RQ5；新增 Stage 4 results 小节；加入 pair-level detection 表和 COCO Caption 辅助表；VQAv2/random 未放入正文。 | 直接回答 RQ5，并保持 downstream 解释为辅助 tradeoff。 | Yes |
| P-DISC-01 至 P-DISC-03 | Discussion 降低 MLLM 绑定；limitations 改为 core pipeline 是 modality-specific，Stage 4 覆盖 image-text case，其他 pair types 未来工作。 | 与新系统边界一致。 | Yes |
| P-CONCL-01 至 P-CONCL-02 | Conclusion 改为 classification-and-clean framework + clean image/audio/text corpora + Stage 4 extension finding。 | 结论跟随正文证据层级。 | Yes |
| P-FIG-01 至 P-FIG-02 | 图本身未处理；正文仅保留现有图引用。 | 用户要求图表暂不处理。 | No |
| P-TITLE-01 | 标题未改。 | 用户要求标题后续细致讨论。 | No |
| P-COMMENT-01 | 旧 comment 未清理。 | 用户要求修改获批后再决定是否清理。 | No |
| P-VERIFY-01 | 编译 `paper/ieee/main.tex` 成功，输出 16 页 `paper/ieee/main.pdf`；无 undefined citation/reference；渲染检查了第 1、8、13、14 页。 | 验证 IEEE 版可编译且新增 Stage 4 内容可读。 | Yes |

当前剩余注意事项：

- 标题仍是旧标题，后续需要单独讨论并降调。
- IEEE PDF 当前为 16 页，后续可能需要压缩篇幅。
- LaTeX log 仍有少量 overfull/underfull box 警告，多数来自原有表格和 IEEE 栏宽；当前不阻塞阅读，但最后定稿前应统一处理。

## 16. 2026-06-03 引用和 Stage 4 图表补充记录

触发原因：

- 用户指出正文中出现了新增引用，要求采用方案一：撤掉非必要新增 active 引用。
- 用户希望 Stage 4 相关实验结果可以补充一些图表。
- 用户再次强调每一处修改都需要在本计划中可追溯。

本轮实际修改文件：

- `paper/ieee/main.tex`
- `paper/ieee/main.pdf`
- `paper/revision_notes/revision_plan.md`

引用处理：

- 已撤掉 active 正文中的 `fang2023data` 和 `evans2024data` 引用。
- `fang2023data` 和 `evans2024data` 仍存在于 `paper/ieee/main.tex` 的 `comment` 旧稿块中，但不参与编译；本轮按“暂不清理 comment”的原则没有删除。
- `paper/ieee/references.bib` 未修改。
- `paper/ieee/main.bbl` 中出现的 `Fang` 是 DataComp 论文作者名，不是 `fang2023data` 引用。

Stage 4 图表处理：

- 新增 `pgfplots` 包，用 LaTeX 内嵌方式绘制 Stage 4 结果图，未新增外部图片文件。
- 新增 Figure：`Stage~4 extension results`。
- 图左侧展示 CC3M-PairEval 上各方法的 pair-level F1：image-only、text-only、naive union、Stage 4 joint、Stage 4 conservative。
- 图右侧展示 A/B/C/D/E splits 的 retained pairs，并叠加 CIDEr 和 BLEU-4，表达 Stage 4 conservative 的 data-retention / caption-transfer tradeoff。
- 图的数据和正文两张 Stage 4 表一致，来源仍是：
  - `experiments/results/plan_b_stage4/exp_stage4_fair_eval_3000_conservative_and_20260601/fixed_threshold_metrics_with_conservative_and.csv`
  - `experiments/results/plan_b_stage4/icdm_revision/summary_20260530/llava_coco_caption_val2014_5k_ckpt1500_20260601.csv`

| 对应位置 | 实际怎么改 | 修改原因 | Applied |
|---|---|---|---|
| Related Work / `Image-Text Dataset Curation` | 将 `"Data Filtering Networks~\\cite{fang2023data} and JEST~\\cite{evans2024data}..."` 改为无新增引用的概括句：image-text curation efforts focus on filtering noisy pairs, selecting useful subsets, or controlling evaluation contamination。 | 执行用户确认的方案一，避免增加非必要 active 引用。 | Yes |
| Stage 4 Results | 在 Stage 4 小节中新增 Figure~`stage4_summary`，用两个面板可视化 direct detection F1 和 downstream split tradeoff。 | Stage 4 结果除表格外需要更直观的图表支撑。 | Yes |
| P-RESULTS-03 至 P-RESULTS-05 | 保留原 pair-level 表和 COCO Caption 表，同时增加图的文字解释：joint pair scoring improves direct duplicate detection; conservative operating point keeps more data for downstream captioning。 | 图、表、正文三者共同支撑 RQ5，但仍保持 conservative tradeoff 说法。 | Yes |
| P-VERIFY-01 | 重新编译 `paper/ieee/main.tex` 成功，输出 16 页 `paper/ieee/main.pdf`；无 undefined citation/reference；检查第 13、14 页渲染。 | 验证撤引用和新增图后 IEEE PDF 仍可编译、可读。 | Yes |

当前注意事项：

- 新增 `pgfplots` 包会增加 TeX 编译依赖；本机 TeX Live 2026 可正常编译。若后续希望减少 LaTeX 包依赖，可以把该图导出为静态 PDF 后再 `includegraphics`。
- Figure~`stage4_summary` 本轮已重新压缩坐标轴宽度并通过 PDF 页面渲染检查；新增图本身不再产生 overfull box。

## 17. 当前写作优先级更新

用户最新确认：

- 目前不需要优先考虑篇幅。
- 主要目标是把 Stage 4 作为支线 extension 插入进来。
- 同时继续调整全文口径，避免过度声明 `MLLM` / `multimodal deduplication`。

后续正文修改优先顺序：

1. 继续检查 active text 中是否仍有过强说法，例如 `multimodal data deduplication framework for MLLM training`、`effective multimodal deduplication requires...`、`industrial MLLM training data preparation` 等。
2. 强化 Stage 4 的支线连贯性：System Design -> Experimental Setup -> Evaluation Results -> Discussion -> Conclusion 必须前后一致。
3. 暂时不因为 IEEE 页数、图表数量或表格占位而删减 Stage 4 内容。
4. 标题仍待后续单独讨论，方向是谦虚、准确、不夸大。

## 18. 2026-06-03 Stage 4 图表美化和 image2 使用边界记录

触发原因：

- 用户提出可以调用 image2 生成更优美的学术图，要求简单呈现实验中我们的优势。
- 用户询问我是否掌握 Stage 4 实验数据。

本轮判断：

- 我确认本地已有 Stage 4 数值数据，主要来自 3,000-label pair-level evaluation 和 COCO Caption 诊断结果。
- 对精确实验图，不采用 image2 直接生成位图，因为坐标轴、数值、legend 和论文表述必须完全可追溯；生成式图片容易产生不可控的数值或文字误差。
- 对概念流程图、pipeline schematic 或视觉摘要，后续可以考虑使用 image2；但当前 Fig. 5 属于实验结果图，应使用真实数据生成。

本轮实际修改文件：

- `paper/ieee/main.tex`
- `paper/ieee/main.pdf`
- `paper/revision_notes/revision_plan.md`

| 对应位置 | 实际怎么改 | 修改原因 | Applied |
|---|---|---|---|
| Fig. 5 / Stage 4 Results | 将左侧面板从单一 F1 展示升级为 Precision、Recall、F1 三组柱状图，方法包括 Image、Text、Union、Joint、Cons.。 | 更清楚呈现 naive union 的高召回低精度，以及 Stage 4 joint 的更平衡优势。 | Yes |
| Fig. 5 / Stage 4 Results | 右侧面板保留 kept pairs 柱状图，并叠加 CIDEr 与 BLEU-4 折线；将坐标轴宽度压缩到 IEEE 双栏安全范围。 | 同时表达 Stage 4 conservative 的保留样本优势和 caption-transfer tradeoff，并消除新增图自身的 overfull。 | Yes |
| Fig. 5 / caption | 更新 caption：明确 Panel (a) 解释 precision--recall tradeoff，Panel (b) 解释 retention / caption-transfer tradeoff。 | 让图的结论与正文 RQ5 叙事一致，避免读者只看到数值而不理解支线意义。 | Yes |
| P-VERIFY-01 | 重新编译 `paper/ieee/main.tex` 成功，输出 16 页 `paper/ieee/main.pdf`；渲染检查第 14 页 Fig. 5；新增图本身无 overfull。 | 验证图表美化后 IEEE PDF 可编译、可读、可追溯。 | Yes |

当前注意事项：

- LaTeX log 仍有少量 overfull/underfull 警告，但来自原有表格或末页栏高，不是本轮新增 Stage 4 图导致。
- `fang2023data` 和 `evans2024data` 仍仅存在于 `comment` 旧稿块中，不参与当前 PDF 编译；后续如果清理 comment，可一并移除这些旧草稿引用。

## 19. 2026-06-03 Stage 4 候选图批量生成记录

触发原因：

- 用户要求先按现有实验数据生成一批 Stage 4 相关图，后续由用户挑选认为合适的图加入文章。

本轮实际修改/新增文件：

- `experiments/scripts/build_stage4_candidate_figures.py`
- `paper/ieee/figures/stage4_candidates/`
- `paper/ieee/figures/stage4_candidates/previews/`
- `paper/ieee/figures/stage4_candidates/README.md`
- `paper/ieee/figures/stage4_candidates/gallery.html`
- `paper/revision_notes/revision_plan.md`

数据来源：

- `experiments/results/plan_b_stage4/exp_stage4_fair_eval_3000_conservative_and_20260601/fixed_threshold_metrics_with_conservative_and.csv`
- `experiments/results/plan_b_stage4/icdm_revision/summary_20260530/llava_coco_caption_val2014_5k_ckpt1500_20260601.csv`
- `experiments/results/plan_b_stage4/exp_stage4_fair_eval_3000_bootstrap_ci_20260531/metrics.json`

生成图列表：

| 文件 | 内容 | 可能用途 |
|---|---|---|
| `01_pair_precision_recall_f1.svg` | Stage 4 pair-level Precision / Recall / F1 grouped bars。 | 主结果图候选，直观看出 joint 的 F1 和平衡性。 |
| `02_pair_f1_confidence_intervals.svg` | F1 bar + 95% CI。 | 如果需要强调稳定差距，可替代或补充主结果表。 |
| `03_precision_recall_tradeoff.svg` | Precision-Recall 平面，气泡大小表示 predicted duplicate rate。 | 展示 naive union 的误报代价和 joint 的平衡点。 |
| `04_error_decomposition_tp_fp_fn.svg` | TP / FP / FN stacked bars。 | 展示 false positive / missed positive 结构。 |
| `05_f1_delta_bootstrap_ci.svg` | joint 相比 naive union 和 image-only 的 F1 delta + bootstrap CI。 | 强调提升幅度和置信区间。 |
| `06_coco_retention_caption_metrics.svg` | kept pairs 柱状图 + CIDEr/BLEU-4 overlay。 | 展示 downstream retention / caption-transfer tradeoff。 |
| `07_dedup_rate_vs_cider.svg` | dedup rate 与 CIDEr scatter。 | 展示 aggressive split D 和 conservative split E 的区别。 |
| `08_kept_pairs_vs_caption_metrics.svg` | kept pairs 与 CIDEr/BLEU-4/ROUGE-L 多指标曲线。 | 展示保留样本量与多个 caption 指标的关系。 |

验证：

- 运行 `python3 experiments/scripts/build_stage4_candidate_figures.py` 成功生成 8 张 SVG。
- 使用系统 Chrome 通过 Playwright 渲染 SVG，并生成 PNG preview 到 `paper/ieee/figures/stage4_candidates/previews/`。
- 已抽查 `01_pair_precision_recall_f1.png`、`03_precision_recall_tradeoff.png`、`06_coco_retention_caption_metrics.png`，确认文字、坐标轴和主要结论可读。

当前状态：

- 这些图只是候选图，尚未加入正文。
- 后续用户挑选后，再决定替换现有 Fig. 5、拆分为多张图，或仅保留部分图作为正文/补充材料。

## 20. 2026-06-03 Stage 4 候选图 paper-style 美化记录

触发原因：

- 用户认为可以先生成 SVG，再用 image2 或类似方式美化，使图至少与文章中原有图风格一致。

本轮判断：

- 我接受“先 SVG 后美化”的方向。
- 对精确数值图，最终版本仍应由可控 SVG/代码生成，避免 image2 改动坐标、数值、legend 或相对比例。
- image2 更适合后续做概念图、pipeline schematic、或作为视觉风格参考；若使用 image2 参考风格，最终仍建议把风格反向落实到 SVG。

本轮实际修改/新增文件：

- `experiments/scripts/build_stage4_candidate_figures.py`
- `paper/ieee/figures/stage4_candidates/paper_style/`
- `paper/ieee/figures/stage4_candidates/paper_style/previews/`
- `paper/revision_notes/revision_plan.md`

本轮实际怎么改：

- 在候选图脚本中新增 `paper_style` 输出分支。
- 参考原文已有 Matplotlib 图风格，调整为：
  - 更接近原文的蓝/红/绿/灰配色；
  - 粗标题、粗轴标签、粗 tick label；
  - 虚线浅灰网格；
  - 图内或图上方 legend；
  - 更接近原图的紧凑论文图比例。
- 新增 7 张 paper-style SVG：
  - `paper_style/01_pair_precision_recall_f1_paper.svg`
  - `paper_style/02_pair_f1_confidence_intervals_paper.svg`
  - `paper_style/03_precision_recall_tradeoff_paper.svg`
  - `paper_style/04_error_decomposition_tp_fp_fn_paper.svg`
  - `paper_style/05_f1_delta_bootstrap_ci_paper.svg`
  - `paper_style/06_coco_retention_caption_metrics_paper.svg`
  - `paper_style/07_dedup_rate_vs_cider_paper.svg`
- 使用系统 Chrome 渲染生成对应 PNG preview：
  - `paper_style/previews/*.png`

| 对应位置 | 实际怎么改 | 修改原因 | Applied |
|---|---|---|---|
| Stage 4 候选图脚本 | 新增 paper-style 绘图函数和输出目录。 | 保留原始候选图，同时生成更接近论文已有图风格的版本。 | Yes |
| `01_pair_precision_recall_f1_paper.svg` | 将 legend 改为顶部横向 legend，避免遮挡 `Cons.` 柱子。 | 主结果图需要清楚展示所有方法，不能让图例遮挡数值。 | Yes |
| `03_precision_recall_tradeoff_paper.svg` | 用 bubble plot 展示 precision-recall tradeoff，并保留 `Joint: balanced` 和 `Union: many positives` 注释。 | 直观表达 Stage 4 joint 与 naive union 的差异。 | Yes |
| `06_coco_retention_caption_metrics_paper.svg` | 用 kept pairs 柱状图叠加 CIDEr/BLEU-4 折线。 | 与原文图风格一致地呈现 downstream tradeoff。 | Yes |

验证：

- 重新运行 `python3 experiments/scripts/build_stage4_candidate_figures.py` 成功。
- 使用系统 Chrome 渲染并检查：
  - `paper_style/previews/01_pair_precision_recall_f1_paper.png`
  - `paper_style/previews/03_precision_recall_tradeoff_paper.png`
  - `paper_style/previews/06_coco_retention_caption_metrics_paper.png`
- 这些图仍未加入正文，等待用户挑选。

## 21. 2026-06-03 Stage 4 候选图选择和图例微调记录

触发原因：

- 用户初步认可 `03_precision_recall_tradeoff_paper` 可加入文章。
- 用户要求修改第二张 downstream 图，避免 `Kept / CIDEr / BLEU-4` 图例遮挡柱形图。

本轮实际修改文件：

- `experiments/scripts/build_stage4_candidate_figures.py`
- `paper/ieee/figures/stage4_candidates/paper_style/06_coco_retention_caption_metrics_paper.svg`
- `paper/ieee/figures/stage4_candidates/paper_style/previews/06_coco_retention_caption_metrics_paper.png`
- `paper/revision_notes/revision_plan.md`

| 对应位置 | 实际怎么改 | 修改原因 | Applied |
|---|---|---|---|
| `03_precision_recall_tradeoff_paper` | 用户初步认可该图作为可加入正文的候选图；本轮未改动该图。 | 该图能直观表达 Stage 4 joint 的 balanced tradeoff 和 naive union 的误报代价。 | Yes |
| `06_coco_retention_caption_metrics_paper` | 将图例从图内上方改为标题下方、绘图区上方的横向图例，并将绘图区整体下移。 | 避免 `Kept Pairs / CIDEr / BLEU-4` 图例遮挡柱形图或折线。 | Yes |

验证：

- 重新运行 `python3 experiments/scripts/build_stage4_candidate_figures.py` 成功。
- 使用系统 Chrome 重新渲染 preview。
- 已检查 `paper_style/previews/06_coco_retention_caption_metrics_paper.png`，图例不再遮挡柱形图。

当前状态：

- `03_precision_recall_tradeoff_paper` 和修改后的 `06_coco_retention_caption_metrics_paper` 仍未加入正文，等待用户确认细节后再插入。

## 22. 2026-06-03 Stage 4 两张图正式加入正文记录

触发原因：

- 用户确认 `03_precision_recall_tradeoff_paper` 和修改后的 `06_coco_retention_caption_metrics_paper` 可以加入文章。
- 用户再次强调所有更改需要在 `revision_plan` 中记录。

本轮实际修改文件：

- `paper/ieee/main.tex`
- `paper/ieee/main.pdf`
- `paper/revision_notes/revision_plan.md`

本轮实际怎么改：

- 将原先 Fig. 5 的内嵌 `pgfplots` 双面板图替换为两张已确认的 paper-style PNG 图：
  - `figures/stage4_candidates/paper_style/previews/03_precision_recall_tradeoff_paper.png`
  - `figures/stage4_candidates/paper_style/previews/06_coco_retention_caption_metrics_paper.png`
- 保留原来的 `\label{fig:stage4_summary}`，因此正文中的 `Figure~\ref{fig:stage4_summary}` 引用无需大改。
- 将 Fig. 5 的两个子图 caption 改为：
  - `(a) Pair-level precision--recall tradeoff.`
  - `(b) Retention and caption-transfer tradeoff.`
- 更新 Fig. 5 总 caption，使其对应现在的两张图：
  - Panel (a) 解释 naive union 通过预测更多 duplicate pairs 提高 recall，但 Stage 4 joint 更平衡。
  - Panel (b) 解释 conservative Stage 4 split 保留更多 image-caption pairs，并维持 near-best caption-transfer performance。
- 因为正文不再使用 `pgfplots`，删除了 `\usepackage{pgfplots}` 和 `\pgfplotsset{compat=1.18}`，减少 IEEE 编译依赖。

| 对应位置 | 实际怎么改 | 修改原因 | Applied |
|---|---|---|---|
| `paper/ieee/main.tex` preamble | 删除 `pgfplots` package 和 compat 设置。 | 替换成外部图后不再需要该依赖。 | Yes |
| Stage 4 Results / Fig. 5 | 用两张 paper-style PNG 子图替换原内嵌 pgfplots 图。 | 用户确认两张图可加入正文；新图更直观、更符合原文风格。 | Yes |
| Fig. 5 caption | 改写为 precision--recall tradeoff 与 retention / caption-transfer tradeoff 两部分。 | caption 需要准确对应新图内容。 | Yes |

验证：

- 重新编译 `paper/ieee/main.tex` 成功，输出 16 页 `paper/ieee/main.pdf`。
- `main.log` 未出现 undefined citation/reference。
- 渲染检查第 14 页，确认 Fig. 5 两张图正常并排显示，第二张图例没有遮挡柱形图。

当前注意事项：

- 当前正文使用的是 PNG preview，SVG 源文件仍保留在 `paper_style/` 中，后续如果需要更高质量矢量图，可以再将 SVG 转 PDF 后替换。
- LaTeX log 中仍有原有表格/页面导致的少量 overfull/underfull 警告，本轮新增 Fig. 5 没有引入新的 undefined 引用问题。

## 23. 2026-06-03 Stage 4 正文图缩小记录

触发原因：

- 用户认为刚加入正文的两张 Stage 4 图偏大，希望参考原文 Fig. 3 的视觉比例稍微缩小。

本轮实际修改文件：

- `paper/ieee/main.tex`
- `paper/ieee/main.pdf`
- `paper/revision_notes/revision_plan.md`

本轮实际怎么改：

- 将 Fig. 5 中两个子图宽度从 `0.49\textwidth` 缩小为 `0.43\textwidth`。
- 将两个子图之间的自动拉伸间距 `\hfill` 改为固定的 `\hspace{0.04\textwidth}`。
- 保留原图文件、caption、`\label{fig:stage4_summary}` 和正文引用不变。

| 对应位置 | 实际怎么改 | 修改原因 | Applied |
|---|---|---|---|
| `paper/ieee/main.tex` / Fig. 5 左子图 | 子图宽度改为 `0.43\textwidth`。 | 缩小 Stage 4 precision-recall 图的版面占比，使其更接近 Fig. 3 的辅助图视觉重量。 | Yes |
| `paper/ieee/main.tex` / Fig. 5 右子图 | 子图宽度改为 `0.43\textwidth`。 | 缩小 Stage 4 downstream 图的版面占比，避免该支线实验图在讨论部分显得过于 dominant。 | Yes |
| `paper/ieee/main.tex` / Fig. 5 子图间距 | 使用 `\hspace{0.04\textwidth}` 控制两图间距。 | 缩小后仍保持两图居中、并排、留白稳定。 | Yes |

验证：

- 重新编译 `paper/ieee/main.tex` 成功，输出 16 页 `paper/ieee/main.pdf`。
- 渲染检查第 14 页，确认 Fig. 5 两张图正常并排显示，整体尺寸比上一版更小，第二张图例仍未遮挡柱形图。
- 本轮没有改动任何引用、实验数值或正文叙事。
