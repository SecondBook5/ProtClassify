# Project structure and conventions

This document summarizes the refactored layout and how artifacts are tracked.

## Directory layout
- `src/protclassify/paths.py` – centralizes repo paths so all modules resolve
  files relative to the project root.
- `src/protclassify/data/` – FASTA parsing and standardized array IO with
  `ArrayMetadata` + `DataTracker`. Saves to `processed_data/` and logs to
  `processed_data/data_tracker.csv`.
- `src/protclassify/features/` – amino-acid composition and dipeptide features
  for sequence-only baselines that run quickly.
- `src/protclassify/models/` – persistence helpers (`registry.py`) and the
  logistic-regression multilabel baseline (`baseline.py`) that pairs with the
  lightweight features.
- `src/protclassify/pipelines/` – orchestration code (currently a baseline
  pipeline that builds features, trains, and writes a submission).
- `src/protclassify/submission/builder.py` – submission generation for CAFA6,
  saved under `submission/` with a manifest at `submission/submission_tracker.csv`.
- `scripts/` – runnable entry points; `train_baseline.py` drives the CAFA6
  baseline end-to-end.
- `utils/` – compatibility wrappers that re-export the new APIs to keep existing
  notebooks running.

## Tracker philosophy
Every saved artifact is accompanied by a manifest row capturing:
- timestamp
- file name
- feature set
- data split
- optimizer/metric (for models)
- version identifier
- human-readable description

This makes it clear **which experiment produced which file** and supports a
clean audit trail when preparing CAFA6 submissions.

## Migration tips
- Prefer importing from `protclassify.*` inside new code.
- Keep legacy scripts working by leaving their imports pointed at `utils/*`
  until you can update them.
- Store intermediate arrays in `processed_data/` and trained models in
  `models/`; do not scatter artifacts in random locations.
