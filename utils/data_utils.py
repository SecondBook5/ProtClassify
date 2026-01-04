#!/usr/bin/env python3
"""Thin compatibility wrappers around the new data tracking utilities.

The refactor moves IO logic into `protclassify.data.tracking`. These
wrappers keep existing notebooks/scripts working while forwarding to the
new implementation.
"""

from __future__ import annotations

from protclassify.data.tracking import ArrayMetadata, DataTracker, load_npy, load_npy_from_tracker, save_npy

__all__ = [
    "ArrayMetadata",
    "DataTracker",
    "load_npy",
    "load_npy_from_tracker",
    "save_npy",
]
