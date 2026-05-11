# Plan B Windows Server Runbook

This runbook treats the Windows RTX 3090 machine as the experiment server. The
Mac is used for smoke tests and code validation only.

## 1. Sync Code

```powershell
git fetch
git checkout codex/plan-b-stage4-pair-dedup
git pull
```

## 2. Prepare Environment

```powershell
uv sync --extra image
```

Quick dependency check:

```powershell
uv run python -c "import torch, open_clip, PIL, sklearn; print(torch.__version__); print(torch.cuda.is_available())"
```

If CUDA is unavailable, fix the PyTorch/CUDA install before running real CC3M
experiments.

## 3. Run Local Smoke Tests

```powershell
uv run python experiments/scripts/smoke_stage4_pair_dedup.py
uv run python experiments/scripts/smoke_stage4_annotation_flow.py
uv run python experiments/scripts/smoke_stage4_evaluation.py
```

These smoke results are engineering checks only. Do not use them in the paper.

## 4. Run Stage 4 on a Small CC3M Subset

Expected input format:

- sidecar directory: `image.jpg` and `image.txt` share the same stem, or
- CSV with `pair_id,image_path,caption` or `pair_id,image_path,caption_path`.

Example sidecar run:

```powershell
uv run python experiments/scripts/run_stage4_pair_dedup.py `
  --input-dir D:\data\cc3m_subset `
  --output-dir experiments\results\plan_b_stage4\exp_stage4_cc3m_subset_YYYYMMDD `
  --backend open_clip `
  --device cuda `
  --batch-size 64 `
  --joint-method concat `
  --tau-cross 0.95 `
  --cache-dir experiments\results\plan_b_stage4\exp_stage4_cc3m_subset_YYYYMMDD\cache `
  --experiment-id exp_stage4_cc3m_subset_YYYYMMDD
```

## 5. Mine Annotation Candidates

```powershell
uv run python experiments/scripts/mine_stage4_candidates.py `
  --input-dir D:\data\cc3m_subset `
  --output-dir experiments\results\plan_b_stage4\exp_stage4_candidates_YYYYMMDD `
  --backend open_clip `
  --device cuda `
  --batch-size 64 `
  --cache-dir experiments\results\plan_b_stage4\exp_stage4_cc3m_subset_YYYYMMDD\cache `
  --method sklearn `
  --signals image,text,joint `
  --top-k 50 `
  --min-similarity 0.70 `
  --max-candidates 50000 `
  --experiment-id exp_stage4_candidates_YYYYMMDD
```

## 6. Build Annotation Sheet

```powershell
uv run python experiments/scripts/build_annotation_sheet.py `
  --candidates-csv experiments\results\plan_b_stage4\exp_stage4_candidates_YYYYMMDD\stage4_candidate_pairs.csv `
  --output-dir experiments\results\plan_b_stage4\exp_stage4_annotation_YYYYMMDD `
  --target-total 1000 `
  --audit-fraction 0.2 `
  --experiment-id exp_stage4_annotation_YYYYMMDD
```

Fill `label` with one of:

- `duplicate`
- `near-duplicate`
- `not-duplicate`

Rows with `needs_audit=1` should be sent to the collaborator for spot-checking.

## 7. Evaluate Labels

If collaborator audit labels are available, adjudicate first:

```powershell
uv run python experiments/scripts/adjudicate_stage4_annotations.py `
  --annotations-csv experiments\results\plan_b_stage4\exp_stage4_annotation_YYYYMMDD\annotation_sheet.csv `
  --output-dir experiments\results\plan_b_stage4\exp_stage4_adjudicated_YYYYMMDD `
  --conflict-policy mark `
  --experiment-id exp_stage4_adjudicated_YYYYMMDD
```

Rows with empty `final_label` require manual resolution before paper numbers.

```powershell
uv run python experiments/scripts/evaluate_stage4_groundtruth.py `
  --annotations-csv experiments\results\plan_b_stage4\exp_stage4_adjudicated_YYYYMMDD\adjudicated_annotations.csv `
  --output-dir experiments\results\plan_b_stage4\exp_stage4_eval_all_YYYYMMDD `
  --score all `
  --thresholds 0.70,0.75,0.80,0.85,0.90,0.95 `
  --experiment-id exp_stage4_eval_all_YYYYMMDD
```

`--score all` evaluates `image`, `text`, `naive_union`, `joint`, and `max` in
one pass. Use the `joint` row as Stage 4 and `naive_union` as the simple
multimodal baseline.

## Source-of-Truth Rule

Do not copy a number into the paper unless it appears in:

- `experiment_ledger.csv`,
- the experiment directory,
- `config.yaml`,
- `metrics.json` or `per_threshold_metrics.csv`,
- `run_manifest.json`.
