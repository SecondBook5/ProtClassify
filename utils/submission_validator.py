"""
Legacy shim to access submission validation helpers from the `protclassify` package.
"""

from protclassify.submission_validator import (
    prepare_predictions,
    to_submission_dataframe,
    write_submission_tsv,
)

__all__ = ["prepare_predictions", "to_submission_dataframe", "write_submission_tsv"]
