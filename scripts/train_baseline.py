"""CLI wrapper to run the lightweight CAFA6 baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

from protclassify.pipelines.baseline import BaselineInputs, run_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline and emit CAFA-style submission")
    parser.add_argument("fasta", type=Path, help="Path to FASTA file containing protein sequences")
    parser.add_argument("labels", type=Path, help="CSV with protein IDs and GO term lists")
    parser.add_argument("--id-column", default="Entry", help="Column in labels CSV that matches FASTA identifiers")
    parser.add_argument(
        "--label-column",
        default="go_terms",
        help="Column containing comma- or pipe-delimited GO terms per protein",
    )
    parser.add_argument("--no-dipeptide", action="store_true", help="Disable dipeptide features for a faster run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_baseline(
        BaselineInputs(
            fasta_path=args.fasta,
            labels_csv=args.labels,
            id_column=args.id_column,
            label_column=args.label_column,
            use_dipeptide=not args.no_dipeptide,
        )
    )
    print(f"Saved model to: {artifacts.training_result.model_path}")
    print(f"F1-micro: {artifacts.training_result.metrics['f1_micro']:.3f}, F1-macro: {artifacts.training_result.metrics['f1_macro']:.3f}")
    print(f"Submission written to: {artifacts.submission_path}")


if __name__ == "__main__":
    main()
