# ProtClassify – CAFA6-ready Protein Function Prediction

A modular, competition-focused codebase for protein function prediction. The
repository now follows a clear "src/" layout, manifest-backed IO helpers, and
submission builders that mirror the CAFA6 format (protein → GO terms and
optional free text).

## Vision
- **CAFA6 alignment:** Build reusable pipelines that can generate the GO term
  predictions and optional free-text descriptions required by the challenge.
- **Research-grade organization:** Source code lives under `src/protclassify`,
  with tracked artifacts for data, models, and submissions.
- **Reproducibility:** Every saved array, model, or submission is logged to a
  manifest CSV to capture provenance (feature set, split, optimizer, version).

## Repository structure
- `src/protclassify/paths.py` – central location for project paths and tracker
  files.
- `src/protclassify/data/tracking.py` – array storage with manifest metadata via
  `ArrayMetadata` and `DataTracker`.
- `src/protclassify/models/registry.py` – model persistence and lookup through
  `ModelRegistry` and `ModelMetadata`.
- `src/protclassify/submission/builder.py` – CAFA-style submission creators with
  `SubmissionBuilder` and `SubmissionMetadata`.
- `utils/` – backward-compatible wrappers re-exporting the refactored modules
  for existing notebooks.
- `processed_data/`, `models/`, `submission/` – tracked artifact locations
  (manifests live alongside outputs).

## Getting started
1. **Install dependencies** (example):
   ```bash
   pip install -r requirements.txt  # or conda env create -f environment.yml
   ```
2. **Import the new helpers**:
   ```python
   from protclassify.data.tracking import DataTracker, ArrayMetadata
   from protclassify.models.registry import ModelRegistry, ModelMetadata
   from protclassify.submission.builder import SubmissionBuilder, SubmissionMetadata
   ```
3. **Track an intermediate array**:
   ```python
   tracker = DataTracker()
   metadata = ArrayMetadata(name="X_train", featureset="esm2", split="train", version="v1", description="ESM2 embeddings")
   tracker.save_array(array, metadata)
   ```
4. **Save and reload a tuned model**:
   ```python
   registry = ModelRegistry()
   meta = ModelMetadata(model_type="xgb", featureset="esm2", optimizer="optuna", scoremetric="f1", version="v1", accuracy=0.67)
   registry.save(model, meta)
   best_model = registry.load_best(metric="accuracy")
   ```
5. **Generate a CAFA-style submission**:
   ```python
   builder = SubmissionBuilder()
   submission_meta = SubmissionMetadata(attempt_number=1, description="Baseline ESM2 + XGBoost")
   builder.from_predictions(y_pred=decoded_labels, entry_df=entry_df, metadata=submission_meta)
   ```

## Next steps toward CAFA6
- Integrate sequence encoders (e.g., ESM/ProtBERT) and ontology propagation.
- Add training scripts under `src/` for reproducible experiments.
- Expand evaluation tooling to compute the CAFA-weighted F1 metrics.

## License
Please ensure compliance with dataset licenses and CAFA rules when distributing
models or predictions.
