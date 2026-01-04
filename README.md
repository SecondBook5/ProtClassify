# ProtClassify — Protein Function Prediction for CAFA6

A professional, reproducible repository for protein function prediction. This refactor organizes prior experimentation into a competition-ready codebase oriented toward **CAFA6** while remaining a polished portfolio project for PhD applications.

## Objectives
- **CAFA6 alignment:** Deliver a pipeline that produces weighted F1–optimized Gene Ontology predictions for MF, BP, and CC.
- **Reproducibility:** Clear environments, deterministic data handling, and tracked artifacts.
- **Professionalism:** Opinionated structure, documentation-first approach, and explicit contribution pathways.

## Repository Layout
- `src/protclassify/` — Library code (data/model/submission utils + path registry). Add CLIs here as the pipeline hardens.
- `data/raw/` — Immutable competition inputs (FASTA, GO mappings, pfam outputs). Keep read-only.
- `data/interim/` — Joined/cleaned feature tables and split definitions (`features/`, `splits/`).
- `data/processed/` — Versioned NumPy artifacts tracked via `data_tracker.csv`.
- `artifacts/embeddings/` — Saved sequence embeddings and embedding notebooks.
- `artifacts/dim_reduction/` — PCA/Lasso reducers and compressed arrays.
- `artifacts/feature_selection/` — Feature selection notebooks, score tables, and descriptor matrices.
- `artifacts/models/` — `final/`, `checkpoints/`, and `studies/` (Optuna/Grid/Search) plus tracker.
- `artifacts/ensembles/` — Stacking/soft-vote outputs and leaderboard-style blends.
- `artifacts/predictions/` & `artifacts/submissions/` — Offline predictions and competition submissions.
- `reports/` — Lightweight CSV summaries/plots safe to version.
- `envs/` — Conda/venv specs for reproducible runs.
- `archive/` — Legacy/colab drops kept for reference.
- `docs/` — Guides, CAFA6 brief, validation checklist, roadmap, and asset registry.
- `notebooks/` — Exploratory research. Promote validated steps into `src/` CLIs.

## Getting Started
1. **Environment:** Use Python 3.10+. Create an isolated environment and install dependencies once `requirements.txt`/`pyproject.toml` is added.
2. **Imports:** Add the repo to `PYTHONPATH` (`export PYTHONPATH=./src:$PYTHONPATH`) or `pip install -e .` once packaging is in place. Legacy `utils.*` imports still work via shims.
3. **Data placement:** Keep raw inputs in `data/raw/`; stage joined features in `data/interim/`; write derived NumPy/Parquet to `data/processed/`.
4. **Tracked arrays:** Save intermediates via `protclassify.data_utils.save_npy` to keep `data/processed/data_tracker.csv` current.
5. **Reproducible runs:** Seed all randomness in training/evaluation scripts. Prefer CLI entry points in `src/` (to be incrementally added) over notebooks for production runs.

## CAFA6-Focused Workflow (High Level)
1. **Sequence embeddings**
   - Compute embeddings (e.g., ESM/ProtBERT/ProtT5) → store in `artifacts/embeddings/`.
2. **Feature curation & dimensionality reduction**
   - Combine sequence embeddings with tabular metadata.
   - Apply PCA, sparse selection (L1/logistic), Bayesian selectors, and VAE compression. Track outputs in `data/processed/` with the tracker.
3. **Model training**
   - Train tuned RF, XGBoost, MLP/TabNet variants on harmonized feature sets.
   - Persist artifacts and Optuna studies under `artifacts/models/` with clear metadata.
4. **Ensembling & calibration**
   - Produce soft-voting and stacking ensembles in `artifacts/ensembles/`. Calibrate probabilities per ontology when needed.
5. **Submission generation & validation**
   - Use `protclassify.submission_validator` + `protclassify.submission_utils` to format GO-term predictions for MF/BP/CC into `artifacts/submissions/`, with GO propagation and the 1,500-term cap enforced.
6. **Evaluation**
   - Track weighted F1 (information accretion) per ontology via `protclassify.metrics`; maintain validation notebooks/scripts in `docs/`.

## Documentation Set
- `docs/cafa6_brief.md` — Competition constraints, evaluation, and submission formatting.
- `docs/reorg_plan.md` — Current refactor roadmap and task backlog.
- `docs/model_registry.md` — Provenance for embeddings/models (add entries as assets are introduced).

## Next Steps
- Wire CLI entrypoints for train/infer/submit that wrap `protclassify` utilities.
- Add dependency manifest and pre-commit hooks (ruff/black/mypy) to enforce consistency.
- Extend validation to include bootstrap CIs and ontology-aware threshold sweeps (use `protclassify.metrics.maximize_f1_over_thresholds`).
- Build automated reports (e.g., experiment tracking summaries) to surface leaderboard-ready checkpoints.

This structure is intended to be incremental: keep exploratory work in notebooks, promote validated code into `src/`, and record every artifact in `docs/` so reviewers—and future you—can follow the reasoning end-to-end.
