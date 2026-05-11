# Plan B Windows Migration Task List

> Purpose: after syncing this branch to the Windows RTX 3090 machine, use this
> document as the execution checklist. The Mac has already completed engineering
> smoke tests; Windows is the experiment server.

## Current Starting Point

- Branch: `codex/plan-b-stage4-pair-dedup`
- Latest validated commit on Mac: `13e9b8b Add Plan B stage4 validation pipeline`
- Mac status before migration:
  - Stage 4 smoke passed.
  - Candidate mining smoke passed.
  - Annotation sheet smoke passed.
  - Adjudication smoke passed.
  - Evaluation smoke passed.
  - `open_clip` CPU smoke passed.

## Migration Goal

The first Windows goal is not to run the final experiment immediately. The goal
is to prove that the Windows machine can execute the same pipeline with CUDA,
real paths, logs, cache files, and source-of-truth outputs.

## Phase 1: Code and Environment Verification

Run:

```powershell
git fetch
git checkout codex/plan-b-stage4-pair-dedup
git pull
uv sync --extra image
```

Check dependencies and CUDA:

```powershell
uv run python -c "import torch, open_clip, PIL, sklearn; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Pass condition:

- `torch.cuda.is_available()` prints `True`.
- GPU name should be RTX 3090 or the expected 3090-class device.

Stop condition:

- If CUDA is `False`, do not run CC3M. Fix PyTorch/CUDA first.

## Phase 2: Local Smoke on Windows

Run:

```powershell
uv run python experiments/scripts/smoke_stage4_pair_dedup.py
uv run python experiments/scripts/smoke_stage4_annotation_flow.py
uv run python experiments/scripts/smoke_stage4_evaluation.py
uv run python experiments/scripts/smoke_stage4_adjudication.py
```

Pass condition:

- All four scripts print `passed`.

Stop condition:

- If any smoke fails, fix it before touching CC3M.

## Phase 3: Prepare a Small Real CC3M Subset

Prepare 1K-5K image-caption pairs first.

Preferred format:

```text
D:\data\cc3m_subset\
  sample_000001.jpg
  sample_000001.txt
  sample_000002.jpg
  sample_000002.txt
```

Alternative CSV format:

```csv
pair_id,image_path,caption
sample_000001,D:\data\cc3m_subset\sample_000001.jpg,a caption here
```

Pass condition:

- At least 1K valid pairs.
- Captions are non-empty.
- Images can be opened by Pillow.

## Phase 4: Stage 4 Real-Backend Small Run

Run Stage 4 on the small subset:

```powershell
uv run python experiments/scripts/run_stage4_pair_dedup.py `
  --input-dir D:\data\cc3m_subset `
  --output-dir experiments\results\plan_b_stage4\exp_stage4_cc3m_5k_smoke_YYYYMMDD `
  --backend open_clip `
  --device cuda `
  --batch-size 64 `
  --joint-method concat `
  --tau-cross 0.95 `
  --cache-dir experiments\results\plan_b_stage4\exp_stage4_cc3m_5k_smoke_YYYYMMDD\cache `
  --experiment-id exp_stage4_cc3m_5k_smoke_YYYYMMDD
```

Pass condition:

- `metrics.json` exists.
- `stage4_keepers.txt` exists.
- `stage4_drops.txt` exists.
- `cache/stage4_embeddings.npz` exists.
- `stderr.log` has no fatal error.

Record:

- Add or verify one row in `experiments/results/plan_b_stage4/experiment_ledger.csv`.

## Phase 5: Candidate Mining on Small Subset

Run:

```powershell
uv run python experiments/scripts/mine_stage4_candidates.py `
  --input-dir D:\data\cc3m_subset `
  --output-dir experiments\results\plan_b_stage4\exp_stage4_candidates_5k_YYYYMMDD `
  --backend open_clip `
  --device cuda `
  --batch-size 64 `
  --cache-dir experiments\results\plan_b_stage4\exp_stage4_cc3m_5k_smoke_YYYYMMDD\cache `
  --method sklearn `
  --signals image,text,joint `
  --top-k 50 `
  --min-similarity 0.70 `
  --max-candidates 50000 `
  --experiment-id exp_stage4_candidates_5k_YYYYMMDD
```

Pass condition:

- `stage4_candidate_pairs.csv` exists.
- Candidate count is non-zero.
- CSV has image/text/joint similarity columns.

If candidate count is too small:

- Lower `--min-similarity` to `0.60`.
- Increase `--top-k` to `100`.

## Phase 6: Build First Annotation Sheet

Run:

```powershell
uv run python experiments/scripts/build_annotation_sheet.py `
  --candidates-csv experiments\results\plan_b_stage4\exp_stage4_candidates_5k_YYYYMMDD\stage4_candidate_pairs.csv `
  --output-dir experiments\results\plan_b_stage4\exp_stage4_annotation_5k_YYYYMMDD `
  --target-total 200 `
  --audit-fraction 0.2 `
  --experiment-id exp_stage4_annotation_5k_YYYYMMDD
```

Pass condition:

- `annotation_sheet.csv` exists.
- Contains label columns.
- Contains `needs_audit`.

Manual work:

- Fill `label` for all rows.
- Send rows with `needs_audit=1` to collaborator.
- Fill `audit_label` when collaborator returns results.

## Phase 7: Adjudicate and Evaluate Small Labels

Adjudicate:

```powershell
uv run python experiments/scripts/adjudicate_stage4_annotations.py `
  --annotations-csv experiments\results\plan_b_stage4\exp_stage4_annotation_5k_YYYYMMDD\annotation_sheet.csv `
  --output-dir experiments\results\plan_b_stage4\exp_stage4_adjudicated_5k_YYYYMMDD `
  --conflict-policy mark `
  --experiment-id exp_stage4_adjudicated_5k_YYYYMMDD
```

Evaluate:

```powershell
uv run python experiments/scripts/evaluate_stage4_groundtruth.py `
  --annotations-csv experiments\results\plan_b_stage4\exp_stage4_adjudicated_5k_YYYYMMDD\adjudicated_annotations.csv `
  --output-dir experiments\results\plan_b_stage4\exp_stage4_eval_all_5k_YYYYMMDD `
  --score all `
  --thresholds 0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95 `
  --experiment-id exp_stage4_eval_all_5k_YYYYMMDD
```

Pass condition:

- `per_threshold_metrics.csv` exists.
- `metrics.json` contains `best_by_score`.
- `joint` and `naive_union` are both present.

Decision point:

- If `joint` is clearly worse than `naive_union`, pause and inspect examples.
- If positive examples are too few, expand data size before judging the method.

## Phase 8: Scale Up

Only after Phase 1-7 pass:

1. Increase CC3M pool to 10K-50K.
2. Mine candidates again.
3. Build 1,000-row annotation sheet.
4. Aim for 200+ positive examples.
5. Run adjudication and evaluation.
6. Only then prepare the main paper table.

## Phase 9: LLaVA Preparation

After Stage 4 evaluation looks plausible, prepare training splits:

- A: raw.
- B: image-only dedup.
- C: text-only dedup.
- D: naive union.
- E: Stage 4.

Do not shrink A/B/C/D/E unless we explicitly decide to change the design.

## Source-of-Truth Rules

Every experiment that may appear in the paper must have:

- `config.yaml`
- `metrics.json` or metric CSV
- `stdout.log`
- `stderr.log`
- `run_manifest.json`
- one row in `experiment_ledger.csv`

Do not copy any number into the paper unless it is traceable to these files.

## Immediate Next Action on Windows

Start with Phase 1 and Phase 2 only. After both pass, move to the 1K-5K CC3M
subset.
