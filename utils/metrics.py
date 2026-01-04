"""
Legacy shim to access metric utilities from the `protclassify` package.
"""

from protclassify.metrics import (
    maximize_f1_over_thresholds,
    threshold_predictions,
    weighted_precision_recall_f1,
)

__all__ = [
    "threshold_predictions",
    "weighted_precision_recall_f1",
    "maximize_f1_over_thresholds",
]
