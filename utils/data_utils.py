#!/usr/bin/env python3
"""Thin compatibility wrappers around the new data tracking utilities.

The refactor moves IO logic into `protclassify.data`. These wrappers keep
existing notebooks/scripts working while forwarding to the new
implementation.
"""
from __future__ import annotations

from protclassify.data import (
    ArrayMetadata,
    DataTracker,
    FastaRecord,
    load_fasta_as_dataframe,
    load_npy,
    load_npy_from_tracker,
    parse_fasta,
    save_npy,
    write_fasta,
)

__all__ = [
    "ArrayMetadata",
    "DataTracker",
    "FastaRecord",
    "load_fasta_as_dataframe",
    "load_npy",
    "load_npy_from_tracker",
    "parse_fasta",
    "save_npy",
    "write_fasta",
]
