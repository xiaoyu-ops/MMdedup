# Stage 4 Annotation App

This local web app is used to label CC3M image-caption candidate pair-pairs for
Plan B Stage 4 evaluation.

## Primary Annotation

Current ICDM fair-evaluation input:

```text
experiments/results/plan_b_stage4/exp_stage4_fair_annotation_3000_20260523/annotation_sheet.csv
```

Run from the repository root:

```bash
uv run python experiments/scripts/serve_stage4_annotation_app.py \
  --annotations-csv experiments/results/plan_b_stage4/exp_stage4_fair_annotation_3000_20260523/annotation_sheet.csv \
  --host 127.0.0.1 \
  --port 8765 \
  --annotator wzy
```

Open:

```text
http://127.0.0.1:8765
```

Output:

```text
experiments/results/plan_b_stage4/exp_stage4_fair_annotation_3000_20260523/annotation_sheet_labeled.csv
```

The app writes to the output CSV immediately after each label click. It does
not overwrite the original annotation sheet.

The old 1,000-row high-joint annotation is now a dev / threshold diagnostic
set only. Do not use it as the final ICDM held-out Stage 4 evaluation set.

## Labels

Use exactly one of:

- `duplicate`
- `near-duplicate`
- `not-duplicate`

Evaluation treats `duplicate` and `near-duplicate` as positive examples.

Score-assisted labeling rule used during the 3000-row annotation pass:

- If both `image_similarity` and `text_similarity` are above `0.85` but below
  `0.95`, the pair can be labeled `near-duplicate`.
- If both `image_similarity` and `text_similarity` are above `0.95`, the pair
  can be labeled `duplicate`.

This is an annotator guideline, not the Stage 4 evaluation threshold. Visual
and caption semantics remain the final basis for the human label.

## Audit Annotation

For collaborator audit, use a separate output file and audit mode:

```bash
uv run python experiments/scripts/serve_stage4_annotation_app.py \
  --annotations-csv experiments/results/plan_b_stage4/exp_stage4_fair_annotation_3000_20260523/annotation_sheet.csv \
  --host 127.0.0.1 \
  --port 8766 \
  --mode audit \
  --annotator collaborator \
  --output-csv experiments/results/plan_b_stage4/exp_stage4_fair_annotation_3000_20260523/annotation_sheet_audit.csv
```

Audit mode only navigates rows where `needs_audit` is true / 1 and writes
labels into `audit_label`.

## Shortcuts

- `1`: duplicate
- `2`: near-duplicate
- `3`: not-duplicate
- left / right arrow: previous / next row
