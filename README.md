# MMdedup

MMdedup is an end-to-end multimodal data cleaning and deduplication framework for building machine learning training corpora from mixed web data. It combines file sorting, modality-specific deduplication, and an ICDM-oriented Stage 4 extension for image-caption pair-level deduplication.

The repository currently serves two purposes:

- A reusable Python codebase for sorting and deduplicating image, audio, and text files.
- A reproducible research workspace for the MMdedup paper revision, including Stage 4 experiment scripts, source-of-truth metrics, dashboard exports, and paper assets.

## What This Project Does

MMdedup addresses a practical problem in multimodal dataset curation: raw crawled corpora usually contain mixed file types, corrupted files, exact duplicates, near duplicates, and duplicated image-caption training units. The system is organized as a staged pipeline:

1. **Stage 1: File sorting**
   Classify raw files into image, audio, text, or unknown buckets and write manifests for downstream stages.

2. **Stage 2: Modality-level deduplication**
   Run specialized deduplication modules for each modality:
   - Image: CLIP/OpenCLIP embeddings plus quality-aware semantic deduplication.
   - Audio: spectrogram fingerprinting and LSH-style candidate retrieval.
   - Text: n-gram/Jaccard or MinHash-style near-duplicate detection.

3. **Stage 3: Reporting**
   Aggregate stage summaries into machine-readable JSON and a Markdown report.

4. **Stage 4: Image-caption pair deduplication**
   Deduplicate image-caption pairs as multimodal training units. Stage 4 encodes images and captions, builds pair-level representations, scores candidate pairs, and evaluates cross-modal operating points against image-only, text-only, and naive-union baselines.

## Current Research Direction

The active revision target is an ICDM 2026 version of the paper. The main research answer is not merely to run three independent single-modality deduplication pipelines side by side. The added Stage 4 evaluates whether image-caption pairs can be deduplicated more effectively as multimodal units.

The current paper-facing Stage 4 workflow uses:

- A CC3M-derived image-caption pool.
- Candidate mining with image, text, and joint similarity signals.
- A 3,000-row score-space stratified held-out annotation set.
- Baselines for image-only, text-only, naive union, and Stage 4 pair-level rules.
- LLaVA downstream validation scripts for A/B/C/D/E training splits.

Experiment numbers that may appear in the paper should be traced through `experiments/results/plan_b_stage4/experiment_ledger.csv` and the mirrored dashboard data under `docs/stage4_dashboard/data/`.

## Repository Layout

```text
the_work_of_dedup/
├── audio/                         # Audio fingerprint and deduplication code
├── image/                         # Image embedding and Q-SemDeDup code
├── text/                          # Text near-duplicate detection code
├── pipelines/                     # Pipeline orchestration and modality runners
│   ├── sorter.py                  # File classification and manifest generation
│   ├── orchestrator.py            # Multistage pipeline execution
│   ├── multimodal_runner.py       # Main pipeline CLI entry point
│   └── stage4_pair_dedup.py       # Image-caption pair-level Stage 4 pipeline
├── experiments/
│   ├── configs/                   # Reproducible example configs
│   ├── scripts/                   # Evaluation, annotation, dashboard, and LLaVA helpers
│   └── results/plan_b_stage4/     # Source-of-truth Stage 4 ledgers and daily logs
├── docs/
│   ├── stage4_dashboard/          # Front-end-readable dashboard data and reports
│   └── reports/                   # Generated research reports
├── paper/                         # Paper drafts, references, figures, and revision notes
├── requirements/                  # Dependency notes
├── pyproject.toml                 # Project metadata and uv dependency groups
└── AGENTS.md                      # Project-specific collaboration and experiment rules
```

## Installation

The project uses `uv` for local development.

```bash
# Base pipeline dependencies
uv sync

# Image support: CLIP/OpenCLIP, PyTorch, Pillow, scikit-learn
uv sync --extra image

# Audio support
uv sync --extra audio

# Text support
uv sync --extra text

# Everything
uv sync --extra all
```

The Stage 4 and LLaVA scripts may require additional GPU-side packages depending on the experiment environment. Paper-facing LLaVA runs were designed for a Windows machine with an RTX 3090 and mirrored back into this repository as source-of-truth records.

## Quick Start

Run the multimodal pipeline with a YAML config:

```bash
uv run python -m pipelines.multimodal_runner \
  --config experiments/configs/stage4_pair_dedup.yaml
```

For the general image/audio/text pipeline, use or adapt the configs in `experiments/configs/`. Many historical configs contain local Windows paths; update those paths before running on a new machine.

Run Stage 4 smoke tests:

```bash
uv run python experiments/scripts/smoke_stage4_pair_dedup.py
uv run python experiments/scripts/smoke_stage4_evaluation.py
```

Regenerate and validate the Stage 4 dashboard data after changing source-of-truth metrics:

```bash
uv run python experiments/scripts/build_stage4_dashboard_data.py
uv run python experiments/scripts/check_stage4_dashboard_consistency.py
```

## Reproducibility Rules

Paper-facing results should never be copied directly from console output into the manuscript. Each result should have:

- An experiment id.
- A config file or command description.
- Hardware and environment information.
- Raw outputs or intermediate artifacts.
- A metrics JSON/CSV.
- A row in `experiments/results/plan_b_stage4/experiment_ledger.csv`.
- A refreshed dashboard export under `docs/stage4_dashboard/data/`.

Large datasets, model checkpoints, generated images, raw predictions, temporary outputs, and machine-specific configs are intentionally ignored by `.gitignore`. The committed files are meant to preserve code, reproducible configuration examples, curated metrics, and paper-facing summaries without shipping large private datasets.

## Important Notes For Open Source Users

- The repository contains historical research artifacts as well as reusable code. Prefer `pipelines/`, `audio/`, `image/`, `text/`, and `experiments/scripts/` for active implementation work.
- Some legacy configs reference local Windows paths. Treat them as templates and replace paths before running.
- Stage 4 evaluation depends on annotation and metric files already mirrored into the repository; raw CC3M images and LLaVA checkpoints are not included.
- Dashboard JSON/CSV files are a project-facing snapshot, not the primary source of truth. The experiment ledger and per-experiment metrics remain authoritative.

## Citation

The paper is under revision. A citation entry will be added after the public manuscript metadata is finalized.
