# CAFA6 Improvement Plan

The repository now has a manifest-driven core (`src/protclassify`) and stable path helpers, but we still need competition-grade plumbing. This plan lists concrete upgrades to make the codebase CAFA6-ready.

## 1) Data/ontology hygiene
- **GO version pinning and cache**: Add a small helper (e.g., `protclassify.data.ontology`) that downloads and caches a specific GO OBO release, records its SHA/date in the manifest, and exposes parent/child lookup tables. Tie the version to every preprocessing run so embeddings and labels are traceable to the ontology snapshot.
- **Slim test superset ingestion**: Build a loader that accepts FASTA IDs + sequences, validates duplicates, and preserves the order expected by the submission file. Emit a manifest entry that records source (challenge test superset vs. local validation sets).
- **Label propagation utilities**: Add GO DAG traversal helpers to propagate predicted terms to ancestors, mirroring CAFA evaluation rules. This should integrate with the submission builder so we can optionally auto-propagate before writing.

## 2) Feature generation and scaling
- **Modern protein embeddings**: Introduce an embedding registry with adapters for ESM-2, ProtT5/ProtBERT, and linear-scaling transformers (e.g., ESM-2-t6) so we can trade accuracy for throughput. Cache embeddings with metadata (model name, layer, pooling strategy) in the data tracker.
- **Sequence-level augmentations**: Support truncation/padding strategies, byte-pair encoding for transformer tokenizers, and optional motif extraction (e.g., k-mer frequencies) as lightweight fallback features.
- **Scaler persistence**: Save fitted scalers (Standard/MinMax/Quantile) into the model tracker with clear linkage to the feature set used to train downstream models.

## 3) Modeling & ensembling
- **Multilabel heads on top of frozen embeddings**: Wrap common classifiers (XGBoost/LightGBM/Linear SVM with calibrated probabilities) in a unified API that logs hyperparameters to the model tracker and saves per-subontology heads.
- **Deep baselines**: Add a simple MLP and a CNN-1D head with configurable class weights and focal loss to handle label imbalance. Use PyTorch Lightning or a minimal training loop that logs metrics per epoch.
- **Ensemble orchestrator**: Create an ensembling module that supports rank averaging, logistic regression stacking, and per-subontology weighting, with validation hooks using held-out sets or cross-validation folds.

## 4) Evaluation parity with CAFA
- **Information-accretion metrics**: Implement the CAFA weighted precision/recall/Fmax calculation locally (with parity tests against the organizers' reference implementation). Provide a CLI entry point (`python -m protclassify.evaluation.run`) that reports MF/BP/CC scores and confidence thresholds.
- **Per-subontology validation splits**: Add utilities to respect label leakage (e.g., ensuring proteins from the same UniRef cluster do not cross splits when possible) and to stratify by subontology coverage.
- **Leaderboard-style sanity checks**: Include a lightweight benchmark script that evaluates on the provided leaderboard subset to detect distribution shifts during development.

## 5) Submission robustness
- **Submission schema validation**: Extend `submission.builder` with schema checks (ID format, GO term validity against the cached ontology, max-1500-term constraint) and automatic tab-separated export with fixed precision.
- **Free-text draft support**: Add optional text-line stubs per protein with confidence scores to prototype the optional text prediction track.
- **Reproducible bundle**: Provide a command that packages the exact models, scalers, ontology snapshot, and submission file into a single artifact for archival and reruns.

## 6) Reproducibility & automation
- **Experiment manifests**: Store run configurations (seed, model, feature set, ontology version) as YAML/JSON next to outputs; register them in trackers for lineage.
- **CLI + Make targets**: Add a `protclassify` CLI (via `python -m protclassify.cli`) with subcommands for `prepare-data`, `embed`, `train`, `evaluate`, and `submit`. Back it with Make targets for one-command reproduction.
- **CI smoke tests**: Wire a minimal CI job that lint-checks (`ruff`), type-checks (`mypy` on `src/protclassify`), and runs fast unit tests on synthetic sequences.

## 7) Documentation
- **Playbooks**: Add short how-tos for (a) preparing the CAFA test superset, (b) embedding at scale on GPU, (c) training and ensembling, and (d) producing a submission file.
- **Model cards**: For each saved model, generate a short markdown card capturing data used, ontology version, and validation metrics.

These steps will give us end-to-end traceability (ontology → features → models → submissions), competitive baselines, and automation to iterate quickly before the CAFA6 deadline.
