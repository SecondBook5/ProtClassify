# Project structure and conventions

This document summarizes the refactored layout and how artifacts are tracked.

## Directory layout
- `src/protclassify/paths.py` – centralizes repo paths so all modules resolve
  files relative to the project root.
- `src/protclassify/data/tracking.py` – standardized array IO with
  `ArrayMetadata` + `DataTracker`. Saves to `processed_data/` and logs to
  `processed_data/data_tracker.csv`.
- `src/protclassify/models/registry.py` – model persistence via
  `ModelRegistry`/`ModelMetadata`, writing to `models/` and
  `models/model_tracker.csv`.
- `src/protclassify/submission/builder.py` – submission generation for CAFA6,
  saved under `submission/` with a manifest at `submission/submission_tracker.csv`.
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
