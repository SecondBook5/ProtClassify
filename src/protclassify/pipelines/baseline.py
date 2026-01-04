"""End-to-end baseline training pipeline for CAFA6 submission prep."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from protclassify.data.fasta import load_fasta_as_dataframe
from protclassify.features.aac import amino_acid_composition, dipeptide_frequencies
from protclassify.models.baseline import ModelMetadata, TrainingResult, predict_proba, train_logreg_baseline
from protclassify.submission.builder import SubmissionBuilder


@dataclass
class BaselineInputs:
    fasta_path: Path
    labels_csv: Path
    label_column: str = "go_terms"
    id_column: str = "Entry"
    use_dipeptide: bool = True


@dataclass
class BaselineArtifacts:
    training_result: TrainingResult
    submission_path: Path


def _prepare_features(df: pd.DataFrame, use_dipeptide: bool) -> pd.DataFrame:
    aac = amino_acid_composition(df["sequence"])
    if use_dipeptide:
        dipep = dipeptide_frequencies(df["sequence"])
        return pd.concat([aac, dipep], axis=1)
    return aac


def _extract_labels(label_series: Iterable[str]) -> list[list[str]]:
    """Parse pipe- or comma-delimited GO terms into lists."""

    parsed: list[list[str]] = []
    for cell in label_series:
        if pd.isna(cell) or not str(cell).strip():
            parsed.append([])
        else:
            tokens = [tok.strip() for tok in str(cell).replace("|", ",").split(",") if tok.strip()]
            parsed.append(tokens)
    return parsed


def run_baseline(inputs: BaselineInputs) -> BaselineArtifacts:
    """Train a sequence-only baseline and emit a CAFA-style submission skeleton."""

    fasta_df = load_fasta_as_dataframe(inputs.fasta_path)
    labels_df = pd.read_csv(inputs.labels_csv)
    merged = labels_df.merge(fasta_df, left_on=inputs.id_column, right_on="protein_id", how="inner")

    X = _prepare_features(merged, use_dipeptide=inputs.use_dipeptide)
    labels = _extract_labels(merged[inputs.label_column])

    training_result = train_logreg_baseline(
        X,
        labels,
        metadata=ModelMetadata(
            name="logreg_baseline",
            featureset="aac_dipep" if inputs.use_dipeptide else "aac",
            description="LogReg OVR baseline over amino-acid compositions",
        ),
    )

    probas = predict_proba(training_result.model, X)
    go_terms = training_result.label_binarizer.classes_

    builder = SubmissionBuilder()
    for protein_id, scores in zip(merged[inputs.id_column], probas):
        for go_term, score in zip(go_terms, scores):
            builder.add_entry(protein_id=str(protein_id), go_term=str(go_term), score=float(score))

    submission_path = builder.save()

    return BaselineArtifacts(training_result=training_result, submission_path=submission_path)


__all__ = ["BaselineInputs", "BaselineArtifacts", "run_baseline"]
