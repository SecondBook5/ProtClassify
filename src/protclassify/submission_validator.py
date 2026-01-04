"""
Validation and formatting helpers for CAFA-style submissions.
"""

from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import pandas as pd

from protclassify import paths
from protclassify.go_graph import GOTermGraph, cap_terms

# Type aliases
GOScoreMap = Mapping[str, float]
TargetPredictions = Mapping[str, GOScoreMap]


def _resolve_allowed_terms(allowed_terms: Optional[Iterable[str]]) -> Set[str]:
    return set(allowed_terms) if allowed_terms is not None else set()


def filter_invalid_terms(
    term_scores: GOScoreMap, allowed_terms: Set[str]
) -> Tuple[Dict[str, float], Set[str]]:
    """
    Filter term scores to allowed GO terms and return any invalid terms.
    """
    if not allowed_terms:
        return dict(term_scores), set()
    filtered = {t: s for t, s in term_scores.items() if t in allowed_terms}
    invalid = set(term_scores) - allowed_terms
    return filtered, invalid


def prepare_predictions(
    predictions: TargetPredictions,
    graph: GOTermGraph,
    allowed_terms: Optional[Iterable[str]] = None,
    min_score: float = 0.0,
    max_terms: int = 1500,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, dict]]:
    """
    Validate, propagate, and cap predictions per target.

    Args:
        predictions: Mapping target_id -> {go_term: score}.
        graph: GO graph for propagation.
        allowed_terms: Optional whitelist of GO terms from the current ontology release.
        min_score: Drop predictions below this score before propagation.
        max_terms: Maximum number of terms per target (CAFA limit: 1500).

    Returns:
        cleaned_predictions: target -> propagated, capped scores
        report: target -> summary dict (counts + invalid terms)
    """
    allowed_set = _resolve_allowed_terms(allowed_terms)
    cleaned: Dict[str, Dict[str, float]] = {}
    report: Dict[str, dict] = {}

    for target, term_scores in predictions.items():
        filtered_scores = {t: s for t, s in term_scores.items() if s >= min_score}
        filtered_scores, invalid_terms = filter_invalid_terms(filtered_scores, allowed_set)

        propagated = graph.propagate_scores(filtered_scores)
        capped = cap_terms(propagated, max_terms=max_terms)

        cleaned[target] = capped
        report[target] = {
            "invalid_terms": sorted(invalid_terms),
            "n_original": len(term_scores),
            "n_after_filter": len(filtered_scores),
            "n_after_propagation": len(propagated),
            "n_after_cap": len(capped),
        }

    return cleaned, report


def to_submission_dataframe(
    predictions: Mapping[str, Mapping[str, float]],
    id_column: str = "Entry",
    term_column: str = "GO_term",
    score_column: str = "score",
) -> pd.DataFrame:
    """
    Convert predictions into a CAFA-style submission DataFrame.
    """
    rows = []
    for target, term_scores in predictions.items():
        for term, score in term_scores.items():
            rows.append({id_column: target, term_column: term, score_column: score})
    return pd.DataFrame(rows, columns=[id_column, term_column, score_column])


def write_submission_tsv(
    predictions: Mapping[str, Mapping[str, float]],
    output_path: Path,
    id_column: str = "Entry",
    term_column: str = "GO_term",
    score_column: str = "score",
) -> Path:
    """
    Write predictions to a tab-separated file without a header (CAFA format).
    """
    df = to_submission_dataframe(predictions, id_column=id_column, term_column=term_column, score_column=score_column)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False, header=False)
    return output_path
