"""Feature extraction utilities for amino acid sequences.

These helpers support CAFA-style inputs where only the primary sequence is
available. They provide simple, fast features that work well with classic
machine learning models and are easy to recompute across new test sets.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

# Standard amino acid vocabulary used by CAFA competitions
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def amino_acid_composition(sequences: Iterable[str]) -> pd.DataFrame:
    """Compute per-sequence amino acid composition features.

    The output contains one column per amino acid plus a sequence length column.
    Values are normalized frequencies in ``[0, 1]`` to make them model-friendly.
    """

    rows: list[dict[str, float]] = []
    for seq in sequences:
        seq = (seq or "").upper()
        counts = Counter(seq)
        length = len(seq) if len(seq) else 1  # avoid division by zero
        rows.append(
            {**{aa: counts.get(aa, 0) / length for aa in AMINO_ACIDS}, "seq_len": float(length)}
        )
    return pd.DataFrame(rows)


def dipeptide_frequencies(sequences: Iterable[str]) -> pd.DataFrame:
    """Compute dipeptide (k=2) frequencies as a richer, still-lightweight signal."""

    dipeptides = [a + b for a in AMINO_ACIDS for b in AMINO_ACIDS]
    rows: list[dict[str, float]] = []
    for seq in sequences:
        seq = (seq or "").upper()
        counts = Counter(seq[i : i + 2] for i in range(len(seq) - 1))
        length = max(len(seq) - 1, 1)
        rows.append({dp: counts.get(dp, 0) / length for dp in dipeptides})
    return pd.DataFrame(rows)


__all__ = ["AMINO_ACIDS", "amino_acid_composition", "dipeptide_frequencies"]
