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
- `src/protclassify/data/` – FASTA parsing (`fasta.py`) and array tracking
  (`tracking.py`) backed by manifest CSVs.
- `src/protclassify/features/` – amino-acid composition and dipeptide features
  for quick baselines when only primary sequences are available.
- `src/protclassify/models/` – trainable components and registries; includes a
  logistic-regression one-vs-rest baseline wired for multilabel GO prediction.
- `src/protclassify/pipelines/` – opinionated pipelines that stitch together
  data loading, feature computation, modeling, and submission generation.
- `src/protclassify/submission/builder.py` – CAFA-style submission creators with
  `SubmissionBuilder` and `SubmissionMetadata`.
- `scripts/train_baseline.py` – CLI entry point to train the baseline and emit a
  submission TSV in one step.
- `utils/` – backward-compatible wrappers re-exporting the refactored modules
  for existing notebooks.
- `data/raw/` – drop the CAFA6 competition download here (see
  `docs/DATASET_DESCRIPTION.md`).
- `processed_data/`, `models/`, `submission/` – tracked artifact locations
  (manifests live alongside outputs).
- `artifacts/legacy/` – quarantined historical notebooks/arrays/models from the
  pre-refactor code; keep new experiments in the tracked locations above.

## Getting started
1. **Install dependencies** (example):
   ```bash
   pip install -r requirements.txt  # or conda env create -f environment.yml
   ```
2. **Train the sequence-only baseline and emit a submission**:
   ```bash
   python scripts/train_baseline.py data/train_targets.fasta data/train_labels.csv \
     --id-column Entry --label-column go_terms
   ```
   The script will:
   - parse the FASTA targets,
   - compute amino-acid composition + optional dipeptide features,
   - train a logistic-regression one-vs-rest classifier,
   - log metrics/manifests under `models/`, and
   - write a CAFA-style submission TSV under `submission/`.
3. **Consume the building blocks in notebooks**:
   ```python
   from protclassify.data import load_fasta_as_dataframe
   from protclassify.features import amino_acid_composition, dipeptide_frequencies
   from protclassify.models.baseline import train_logreg_baseline
   from protclassify.pipelines import run_baseline
   ```
4. **Track an intermediate array**:
   ```python
   from protclassify.data import ArrayMetadata, DataTracker
   tracker = DataTracker()
   metadata = ArrayMetadata(name="X_train", featureset="esm2", split="train", version="v1", description="ESM2 embeddings")
   tracker.save_array(array, metadata)
   ```
5. **Generate a CAFA-style submission from predictions**:
   ```python
   from protclassify.submission.builder import SubmissionBuilder, SubmissionMetadata
  builder = SubmissionBuilder()
  submission_meta = SubmissionMetadata(attempt_number=1, description="Baseline ESM2 + XGBoost")
  builder.from_predictions(y_pred=decoded_labels, entry_df=entry_df, metadata=submission_meta)
  ```

## Data intake for CAFA6
- Download the Kaggle CAFA6 bundle and place the raw files (e.g.,
  `train_sequences.fasta`, `train_terms.tsv`, `IA.tsv`) under `data/raw/`.
- See `docs/DATASET_DESCRIPTION.md` for a concise recap of the files provided by
  the competition and how they relate to our pipelines.

## Next steps toward CAFA6
- Integrate sequence encoders (e.g., ESM/ProtBERT) and ontology propagation.
- Add training scripts under `src/` for reproducible experiments.
- Expand evaluation tooling to compute the CAFA-weighted F1 metrics.
- Follow the detailed roadmap in `docs/CAFA6_IMPROVEMENT_PLAN.md` for
  data hygiene, modeling, evaluation parity, and submission hardening.

## License
Please ensure compliance with dataset licenses and CAFA rules when distributing
models or predictions.
