# CAFA6 dataset overview and intake

This repository expects the CAFA6 competition bundle (Kaggle: `cafa-6-protein-function-prediction`) to be placed under `data/raw/`. The bundle provides:

- **Ontology**: `go-basic.obo` (2025-06-01 release) – the GO DAG covering MF/BP/CC with roots `GO:0003674`, `GO:0008150`, `GO:0005575`.
- **Training set**:
  - `train_sequences.fasta` – Swiss-Prot sequences (UniProt 2025_03, June 18 2025) for proteins that have at least one experimentally supported GO term.
  - `train_terms.tsv` – tab-separated ground-truth GO term labels for the training proteins.
  - `train_taxonomy.tsv` – UniProt accession → NCBI taxon ID mapping for the training proteins.
  - `IA.tsv` – information accretion weights for GO terms used by the CAFA weighted-F1 metric.
- **Test superset**:
  - `testsuperset.fasta` – sequences for proteins that require predictions (subset will be scored once new annotations appear).
  - `testsuperset-taxon-list.tsv` – taxon IDs for proteins in the test superset.
- **Reference**:
  - `sample_submission.tsv` – example CAFA-format output.

## Placement
1. Download the dataset locally (e.g., `kaggle competitions download -c cafa-6-protein-function-prediction`).
2. Extract all files into `data/raw/`.
3. Point training/evaluation scripts to the raw FASTA/TSV paths in that directory.

## Notes for modeling
- GO annotations are **positive-only**; absence of a term does not imply negative evidence.
- Evidence codes are restricted to experimental/high-throughput/TAS/IC in `train_terms.tsv`.
- Weighted metrics rely on `IA.tsv`; wire it into evaluation to mirror the official leaderboard.
