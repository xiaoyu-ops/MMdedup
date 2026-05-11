# Plan B Stage 4 Experiment Records

This directory is the source of truth for all CIKM Plan B / Stage 4 experiments.

Do not cite a number in the paper unless it is traceable to:

1. a row in `experiment_ledger.csv`,
2. an experiment directory named by `experiment_id`,
3. a saved config,
4. raw outputs or annotations,
5. computed metrics.

## Directory Layout

```text
experiments/results/plan_b_stage4/
  README.md
  experiment_ledger.csv
  daily_logs/
  <experiment_id>/
    config.yaml
    metrics.json
    stdout.log
    stderr.log
```

## Current Main Experiments

- Stage 4 CLIP joint embedding pair dedup.
- CC3M candidate pair-pair mining and annotation.
- Stage 4 vs image-only / text-only / naive union.
- LLaVA A/B/C/D/E downstream comparison.
- Stage 4 efficiency / overhead.

