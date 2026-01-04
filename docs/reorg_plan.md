# Refactor Roadmap

This roadmap turns the legacy exploratory repo into a maintainable, CAFA6-ready project. Tackle items in order of impact and dependency.

## 1) Foundations
- [ ] Generate `requirements.txt` or `pyproject.toml` pinned to reproducible versions.
- [ ] Introduce formatting/linting (`ruff`, `black`, `isort`) and type checks (`mypy`) via pre-commit.
- [x] Stand up `src/protclassify` package with shared utilities and path registry (CLIs next).
- [x] Expand `.gitignore` to protect large artifacts and competition data.

## 2) Data & Experiment Tracking
- [x] Standardize raw/processed directories (`data/raw`, `data/interim`, `data/processed/`).
- [ ] Enforce artifact logging using `utils/data_utils.py` with run metadata (seed, split, ontology focus).
- [ ] Create `docs/experiments/` with run cards (config, metrics, checkpoints, submission hash).

## 3) Modeling
- [ ] Port validated notebook code into reusable modules (feature engineering, model training, ensembling).
- [ ] Add ontology-specific heads and calibration routines.
- [x] Implement evaluation utilities for weighted F1 (bootstrap CIs next).

## 4) Inference & Submission
- [ ] Build a deterministic inference pipeline that loads embeddings → features → models → ensembles.
- [x] Harden submission utils with GO propagation + 1,500-term caps.
- [ ] Provide sample submission generation script with CLI flags.

## 5) Observability
- [ ] Integrate lightweight experiment tracking (CSV logs + optional MLflow/Weights & Biases).
- [ ] Add automated report generation summarizing per-ontology performance and calibration.

## 6) Documentation
- [ ] Keep `docs/model_registry.md` updated with model/embedding sources and licenses.
- [ ] Expand README with reproducible run commands once CLIs are in place.
- [ ] Add architecture diagram illustrating data flow from embeddings to submissions.

This plan keeps exploratory work in notebooks while promoting stable components into versioned code, ensuring the repository presents well for PhD applications and is competitive for CAFA6.
