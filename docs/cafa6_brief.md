# CAFA6 Competition Brief

**Goal:** Predict Gene Ontology (GO) terms for provided protein sequences and maximize information-accretion weighted F1 across the three ontologies: Molecular Function (MF), Biological Process (BP), and Cellular Component (CC).

## Key Requirements
- **Submission format:** Tab-separated lines with `ProteinID\tGO:Term\tScore`. No header. Scores in `(0, 1]` with up to three significant figures. Maximum of 1,500 terms per protein across all ontologies.
- **Propagation rule:** Parent GO terms are assigned the max score of their children if not provided. Invalid GO terms are dropped during evaluation.
- **Optional text task:** Up to five free-text lines per protein using `ProteinID\tText\tScore\tDescription` with ASCII printable characters only.
- **Evaluation:** Weighted precision/recall (information accretion) → maximum F1 per ontology → averaged across MF/BP/CC. Prospective evaluation: test set is determined after submission based on new experimental annotations.

## Recommended Pipeline
1. **Embeddings**: Generate sequence embeddings (e.g., ESM/ProtBERT); log provenance in `docs/model_registry.md`.
2. **Feature Engineering**: PCA, L1-based selection, Bayesian selectors, VAEs. Track derived arrays via `utils/data_utils.py` to populate `processed_data/data_tracker.csv`.
3. **Model Training**: Tune RF, XGBoost, MLP, and TabNet on unified feature matrices. Retain Optuna studies and final models in `final_models/` with metadata.
4. **Ensembling**: Soft-voting and stacking (e.g., LightGBM) to balance calibration vs. accuracy. Consider ontology-specific heads if performance diverges.
5. **Validation**: Weighted F1 by ontology on held-out splits; bootstrap confidence intervals to gauge stability. Use MC dropout or temperature scaling for calibrated probabilities.
6. **Submission Generation**: Use `utils/submission_utils.py` to produce submission-ready TSVs respecting the 1,500-term cap and score precision.

## Checklist Before Submission
- [ ] Verify ontology versions match competition release.
- [ ] Confirm all predictions are propagated to ontology roots or allow automatic propagation.
- [ ] Ensure deterministic seeds across training and inference.
- [ ] Document training commands, seeds, and data sources in run logs under `docs/`.
- [ ] Perform sanity checks against leaderboard sample proteins where available.

## Long-Term Opportunities
- Incorporate protein–protein interaction graphs or expression data for context-aware predictions.
- Explore contrastive pretraining on PFAM/unlabeled sequences to enrich embeddings.
- Add textual description generation for the optional free-text track (retain evidence IDs when possible).
