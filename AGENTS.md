# AGENTS.md

> Last updated: 2026-05-10
> Branch: `codex/plan-b-stage4-pair-dedup`
> Current target: CIKM 2026 Full Paper

This file is the working handoff for Codex and human collaborators on the Plan B
revision of MMdedup. It records the current thesis, stage goals, required
artifacts, and experiment source-of-truth rules.

## Current Direction

We are pausing the Plan A branch (`feat/text-qsemdedup`) and returning to a
more focused Plan B from `main`.

The key reviewer criticism is:

> The original "multimodal" system is mostly three single-modality dedup
> pipelines placed side by side.

The Plan B answer is not to repackage the existing modules as a unified quality
framework. The answer is to add a genuinely cross-modal stage:

> Stage 4: image-caption pair-level deduplication with CLIP joint embeddings.

The paper should argue that MMdedup now performs file-level / modality-level
deduplication first, then removes duplicate image-caption pairs as multimodal
training units.

## Target Claim

Use a precise, defensible claim:

> We introduce a pair-level cross-modal deduplication stage for image-caption
> training corpora and integrate it into an end-to-end multimodal data cleaning
> pipeline for MLLM training.

Avoid risky claims such as:

- "first multimodal deduplication framework for MLLM training"
- "first to deduplicate MLLM data"
- "novel algorithms for all modalities"

## Required Final Artifacts

### 1. Stage 4 Implementation

Deliverable:

- A runnable cross-modal pair dedup stage.

Inputs:

- Image-caption pairs, identified by pair id / shared stem.
- Image path and caption text.

Core method:

- Encode image with CLIP image encoder: `e_img`.
- Encode caption with CLIP text encoder: `e_txt`.
- Build joint embedding:
  - Primary: concat, `[e_img; e_txt]`.
  - Optional ablation: weighted sum, `alpha * e_img + (1 - alpha) * e_txt`.
- Search / cluster in joint embedding space.
- Mark two image-caption pairs as duplicates when joint similarity exceeds
  `tau_cross`.
- When duplicate image-caption pairs are detected, keep the pair with higher
  CLIP image-text alignment. If alignment scores are effectively tied, use image
  quality (resolution / file size) as the tie-breaker.

Outputs:

- `keepers`: retained pair ids.
- `drops`: removed pair ids.
- `duplicate_groups`: keeper -> removed duplicates.
- `summary.json`: counts, thresholds, method config, runtime.
- Embedding cache for reproducibility.

Minimum success condition:

- Stage 4 can run on a small CC3M subset and produce deterministic keep/drop
  outputs.

### 2. CC3M Ground Truth Dataset

Deliverable:

- A real-data evaluation set based on CC3M, replacing the old purely synthetic
  duplicate benchmark.

Important correction:

- Do not randomly sample pair-pairs from CC3M. Random sampling will produce too
  few duplicates.
- First mine likely duplicate candidate pair-pairs, then annotate them.

Pipeline:

1. Download / prepare a CC3M pool, ideally 100K-300K image-caption pairs.
2. Mine candidate pair-pairs using one or more signals:
   - image pHash similarity,
   - CLIP image similarity,
   - CLIP text similarity,
   - CLIP joint similarity.
3. Stratified sample annotation candidates across high / medium / low similarity
   buckets.
4. Human annotate candidate pair-pairs.

Labels:

- `duplicate`: image and caption are both duplicate.
- `near-duplicate`: semantically duplicate with visual transform or caption
  rewrite.
- `not-duplicate`: not a duplicate.

Target:

- At least 1,000 labeled pair-pairs for evaluation.
- At least 200 positive examples (`duplicate` + `near-duplicate`) if possible.
- Record annotator identity and adjudication when labels disagree.

Quality control:

- Primary annotation will be performed by the user.
- Keep a collaborator audit mechanism: sample a subset for collaborator review
  (for example 20%) and record disagreements.
- Report agreement / audit statistics when collaborator review is available.
- If agreement is below 0.6, refine annotation guidelines before continuing.

### 3. Main Stage 4 Evaluation

Deliverable:

- A table proving Stage 4 is better than modality-wise composition.

Baselines:

- Image-only dedup.
- Text-only dedup.
- Naive multimodal baseline: image-only drops union text-only drops.
- Stage 4 cross-modal pair dedup (ours).

Metrics:

- Precision.
- Recall.
- F1.
- Dedup rate.
- Runtime / throughput.

Primary success condition:

- Stage 4 F1 should improve over naive multimodal baseline.
- A meaningful target is absolute F1 improvement of at least 0.03-0.05.
- If Stage 4 F1 is below 0.5 on the labeled ground truth, pause and revisit
  representation choice (for example BLIP / SigLIP) or thresholding.

### 4. MLLM Downstream Validation

Deliverable:

- LLaVA fine-tuning comparison showing deduped data preserves downstream
  performance while reducing training data / training cost.

Target design:

- A: raw data, no dedup.
- B: image-only dedup.
- C: text-only dedup.
- D: image + text independent dedup union (naive multimodal).
- E: Stage 4 cross-modal pair dedup (ours).

Do not shrink this design by default. The A/B/C/D/E comparison is important
because it separates single-modality effects from the naive multimodal union
and the proposed Stage 4 contribution. If time pressure becomes critical, any
reduction of this design must be discussed explicitly before changing the
experiment plan.

Model:

- LLaVA-1.5-7B LoRA fine-tuning.

Evaluation:

- VQAv2.
- TextVQA if time allows.

Success condition:

- Stage 4 uses fewer samples than raw data.
- Stage 4 does not significantly degrade VQAv2 / TextVQA.
- Ideally Stage 4 matches or improves raw and beats naive multimodal.

### 5. System and Efficiency Results

Deliverable:

- A compact table showing Stage 4 overhead is acceptable.

Track:

- CLIP embedding time.
- Pair search / clustering time.
- End-to-end Stage 4 wall-clock time.
- GPU peak memory.
- Throughput per 1K or 100K pairs.

### 6. Paper Revision

Deliverable:

- Revised paper centered on Stage 4 cross-modal pair dedup.

Required edits:

- Rewrite introduction around the "three single-modality pipelines" criticism.
- Add Stage 4 method section.
- Remove or rewrite the self-defeating Section 2.6 phrasing:
  "Rather than proposing novel algorithms..."
- Add CC3M ground truth construction section.
- Add LLaVA downstream validation.
- Add related work discussion for SemDeDup, DataComp, FairDeDup, and SSCD.
- Discuss pHash / MD5 / MFCC favorable results honestly as boundary cases.

## CIKM Full Timeline

CIKM 2026 Full Paper submission deadline: 2026-05-23 AoE.

### Checkpoint 1: 2026-05-12

Goal:

- Stage 4 minimum implementation runs on a small local subset.

Must have:

- CLIP image/text embedding cache.
- Joint embedding construction.
- Similarity thresholding.
- Keep/drop output.

### Checkpoint 2: 2026-05-15

Goal:

- CC3M candidate mining and annotation sheet ready.

Must have:

- Candidate pair-pair CSV.
- Similarity scores from at least one image signal and one cross-modal signal.
- Annotation instructions.

### Checkpoint 3: 2026-05-17

Goal:

- Main Stage 4 evaluation table.

Must have:

- At least several hundred labeled pair-pairs.
- Stage 4 vs image-only / text-only / naive baseline.
- Precision / Recall / F1.

Decision:

- If Stage 4 does not improve over naive baseline, the CIKM Full story is in
  danger and should be reconsidered.

### Checkpoint 4: 2026-05-20

Goal:

- LLaVA pilot progress.

Target:

- Prepare all five training splits A/B/C/D/E.
- Run or queue all five LLaVA LoRA jobs with identical hyperparameters.
- Have at least the first downstream metric available for enough jobs to judge
  whether Stage 4 is plausible.

Do not reduce the A/B/C/D/E design by default. If not all five jobs can finish
by this checkpoint, keep the design intact and record which jobs are still
running or blocked.

### Checkpoint 5: 2026-05-22

Goal:

- Paper frozen except final polishing.

Must have:

- All tables traceable to experiment logs.
- No unverified numbers in text.
- No new experiments added after this point unless replacing failed numbers.

## Daily Progress Log

Every workday must end with a short progress record. This is mandatory because
the deadline is tight and multiple people may touch code, experiments, and the
paper.

Recommended location:

- `experiments/results/plan_b_stage4/daily_logs/`

Recommended file name:

- `YYYY-MM-DD.md`

Each daily log should include:

- What changed today.
- Experiments started / finished.
- Result files and experiment ids created today.
- Current blockers.
- Tomorrow's first priority.
- Any decisions made with the user / collaborators.

Do not rely on chat history as the only progress record.

## Execution Environment Assumptions

Most final experiments and LLaVA training are expected to run on a Windows
machine with one RTX 3090 GPU.

Main-experiment rule:

- Main experiments should run on the Windows RTX 3090 machine by default. This
  includes CC3M-scale candidate mining, Stage 4 full-scale runs, A/B/C/D/E split
  generation when it depends on Windows-side data, efficiency measurements, and
  all LLaVA LoRA training/evaluation jobs.
- The Mac should mainly be used for smoke tests, small deterministic
  validation, code edits, result inspection, source-of-truth consolidation,
  dashboard generation, and paper writing.
- If a paper-facing result is produced on Mac instead of Windows, record the
  reason in the experiment ledger and daily log, and do not mix its runtime or
  efficiency numbers with Windows RTX 3090 results.

Current development machine:

- macOS on Apple M5 Pro with 48 GB memory.

Current data source decision:

- Download / stream CC3M from HuggingFace unless an existing CC3M TSV/image
  cache is provided later.

Current modality focus:

- CIKM Plan B focuses on image-caption cross-modal Stage 4.
- Audio is retained as existing system capability but should not receive major
  new work or become a main experiment for this submission.

Engineering requirements:

- Keep scripts Windows-compatible.
- Avoid POSIX-only shell assumptions in experiment drivers.
- Prefer Python scripts over shell pipelines for reusable experiment workflows.
- Use `pathlib.Path` for filesystem paths.
- Avoid hardcoded macOS-only paths in committed configs.
- Provide portable YAML configs where possible.
- Keep GPU memory in mind: RTX 3090 has 24 GB VRAM.

LLaVA training assumptions:

- Target hardware: single RTX 3090, 24 GB.
- Use LoRA / QLoRA-style fine-tuning rather than full fine-tuning.
- Expect small batch size with gradient accumulation.
- Save checkpoints and logs frequently because long Windows training runs can
  be interrupted.
- Record exact CUDA / PyTorch / driver / GPU information in the experiment
  ledger.

## Experiment Source-of-Truth Rules

This is mandatory. The previous submission was criticized for inconsistent
numbers across the abstract, introduction, tables, and discussion. Therefore,
every experiment must create source-of-truth records before its results can be
used in the paper.

Every experiment that may appear in the paper must have a record with:

- experiment id, e.g. `exp_stage4_threshold_20260512`;
- dataset name and version, e.g. `CC3M-100K-v1`;
- git commit hash;
- hardware configuration;
- full config / hyperparameters;
- wall-clock runtime;
- GPU peak memory if applicable;
- paths to intermediate outputs;
- paths to final metric files;
- exact numbers cited in the paper.

Recommended location:

- `experiments/results/plan_b_stage4/`
- For experiments executed on the Windows RTX 3090 machine, copy/sync the
  resulting experiment directory back to the Mac after the run. The Mac-side
  mirror location is:
  `experiments/results/plan_b_stage4/windows_sync/`.

Recommended files:

- `experiment_ledger.csv`
- `README.md`
- one config YAML per run
- one metrics JSON/CSV per run

Hard rules:

- Do not paste numbers into the paper unless they can be traced to the ledger.
- Windows-generated results must have a Mac-side copy before they are treated
  as stable source-of-truth records. Keep the Windows original and the Mac copy
  until the paper submission is complete.
- Do not report "best" numbers without saving the threshold / config sweep that
  produced them.
- Do not overwrite result files from previous runs; create a new experiment id.
- When a table changes, update the ledger entry or create a new one.
- If a result is hand-curated or manually labeled, store the raw annotation file,
  adjudicated labels, and the script used to compute metrics.

Minimum source-of-truth files:

- `experiments/results/plan_b_stage4/experiment_ledger.csv`
- `experiments/results/plan_b_stage4/README.md`
- `experiments/results/plan_b_stage4/<experiment_id>/config.yaml`
- `experiments/results/plan_b_stage4/<experiment_id>/metrics.json`
- `experiments/results/plan_b_stage4/<experiment_id>/stdout.log`
- `experiments/results/plan_b_stage4/<experiment_id>/stderr.log`

## Stage 4 Dashboard Rules

The project dashboard is a lightweight front end for current progress, required
Plan B data, experiment status, annotation status, and source-of-truth exports.

Dashboard location:

- `docs/stage4_dashboard/`

Dashboard data generator:

- `experiments/scripts/build_stage4_dashboard_data.py`

Required update rule:

- After every experiment that updates the ledger, metrics, annotation status, or
  stage progress, rerun:
  `uv run python experiments/scripts/build_stage4_dashboard_data.py`
- The dashboard must be updated in real time with the current plan progress.
  Any change to experiment status, required Plan B data, annotation progress,
  paper-facing numbers, source-of-truth files, blockers, or next steps must be
  reflected in `docs/stage4_dashboard/data/` before the work is considered
  complete for that step.
- Treat `docs/stage4_dashboard/data/status.json` as the current front-end
  snapshot.
- The dashboard must expose plan-relevant data through front-end-readable files
  under `docs/stage4_dashboard/data/`, including:
  - `status.json`
  - `plan_requirements.json`
  - `latest_annotation_status.json`
  - `experiment_ledger.csv`

Hard rules:

- Do not use the dashboard as the source of truth. The source of truth remains
  the experiment ledger, metrics/config/log files, annotation CSVs, and daily
  logs.
- Do use the dashboard as the public/project-facing view of the current
  source-of-truth state.
- Do not report a stage as done to the user if the website/dashboard still shows
  the old state. First update the source-of-truth files, regenerate dashboard
  data, and verify that the website-facing files expose the new numbers.
- When the dashboard is deployed, push or deploy dashboard changes promptly
  after meaningful progress so collaborators can inspect the same current data
  from the website.
- If the dashboard and ledger disagree, update the ledger or metrics first, then
  regenerate the dashboard.
- If the dashboard is deployed under a domain, deploy the full
  `docs/stage4_dashboard/` directory so the JSON/CSV exports remain available.

Required paper-number workflow:

1. Run experiment and save raw outputs.
2. Compute metrics into `metrics.json` or `metrics.csv`.
3. Add / update one row in `experiment_ledger.csv`.
4. Regenerate the dashboard:
   `uv run python experiments/scripts/build_stage4_dashboard_data.py`
5. Only then copy the number into the paper table.
6. In the paper working notes, cite the experiment id next to each table row.

## Current Priority Order

1. Implement Stage 4 minimum runnable pipeline.
2. Build CC3M candidate mining and annotation sheet.
3. Get Stage 4 vs naive baseline P/R/F1.
4. Run LLaVA minimum downstream validation.
5. Add efficiency table.
6. Write and polish CIKM full paper.

## Deferred / Non-Blocking Items

These are useful but must not block the core CIKM Full path:

- Second-seed repeats for all LLaVA configurations, if the first-seed A/B/C/D/E
  comparison is not yet complete.
- SSCD on all legacy image benchmarks.
- Audio CLAP upgrade.
- SBERT text Q-SemDeDup.
- Unified quality framework from Plan A.
- Large ablation grid.
