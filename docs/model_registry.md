# Model & Embedding Registry

Document every reusable asset with provenance, licensing, and intended use. Keep this file updated as models are added or refreshed.

| Asset | Type | Source/Version | Input Features | Target Ontology | Notes |
| --- | --- | --- | --- | --- | --- |
| _example: esm2_t33_650M_UR50D_2024_01_15_1_0.pt_ | Sequence embedding | HuggingFace `facebook/esm2_t33_650M_UR50D` | Amino acid sequence | All (MF/BP/CC) | Stored under `artifacts/embeddings/`; see SHA256: `<hash>` |
| _example: xgb_full_features_v1.joblib_ | Classifier | XGBoost 1.x | Tabular + embeddings | MF/BP/CC heads | Tuned via Optuna study `artifacts/models/studies/xgb_optuna_study_full.joblib` |

## Logging Checklist
- Download source URL + commit hash/model card.
- Preprocessing steps required (tokenization, padding, scaling, etc.).
- Training data provenance and date range.
- Hyperparameters and seed.
- Metric summary and validation split description.
- SHA256 hash of stored artifact.
