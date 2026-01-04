"""
Legacy shim that forwards to `protclassify.data_utils`.
"""

from protclassify.data_utils import load_npy, load_npy_from_tracker, save_npy

__all__ = ["load_npy", "load_npy_from_tracker", "save_npy"]
