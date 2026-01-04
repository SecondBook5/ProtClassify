"""
Legacy shim to access GO graph helpers from the `protclassify` package.
"""

from protclassify.go_graph import GOTermGraph, cap_terms

__all__ = ["GOTermGraph", "cap_terms"]
