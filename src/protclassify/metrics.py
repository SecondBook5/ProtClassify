"""
Evaluation utilities for CAFA-style weighted precision/recall/F1.
"""

from typing import Dict, Iterable, List, Mapping, Sequence, Set


def threshold_predictions(
    y_pred_scores: Sequence[Mapping[str, float]], threshold: float
) -> List[Set[str]]:
    """
    Convert score dictionaries to sets using a probability threshold.
    """
    pred_sets: List[Set[str]] = []
    for scores in y_pred_scores:
        pred_sets.append({term for term, score in scores.items() if score >= threshold})
    return pred_sets


def weighted_precision_recall_f1(
    y_true: Sequence[Set[str]],
    y_pred_scores: Sequence[Mapping[str, float]],
    term_weights: Mapping[str, float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute information-accretion weighted precision/recall/F1 as in CAFA.

    Args:
        y_true: Sequence of gold term sets.
        y_pred_scores: Sequence of mappings term -> score.
        term_weights: Mapping term -> information accretion weight (ic(f)).
        threshold: Score cutoff for positive prediction.

    Returns:
        Dict with weighted precision, recall, and F1.
    """
    assert len(y_true) == len(y_pred_scores), "Mismatched true/pred lengths"

    preds = threshold_predictions(y_pred_scores, threshold)
    tp_weight = 0.0
    fp_weight = 0.0
    fn_weight = 0.0

    for truth, pred in zip(y_true, preds):
        tp = truth & pred
        fp = pred - truth
        fn = truth - pred

        tp_weight += sum(term_weights.get(t, 0.0) for t in tp)
        fp_weight += sum(term_weights.get(t, 0.0) for t in fp)
        fn_weight += sum(term_weights.get(t, 0.0) for t in fn)

    precision = tp_weight / (tp_weight + fp_weight) if (tp_weight + fp_weight) > 0 else 0.0
    recall = tp_weight / (tp_weight + fn_weight) if (tp_weight + fn_weight) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def maximize_f1_over_thresholds(
    y_true: Sequence[Set[str]],
    y_pred_scores: Sequence[Mapping[str, float]],
    term_weights: Mapping[str, float],
    thresholds: Iterable[float],
) -> Dict[str, float]:
    """
    Sweep thresholds and return the best weighted F1 and corresponding precision/recall.
    """
    best = {"threshold": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for thr in thresholds:
        scores = weighted_precision_recall_f1(y_true, y_pred_scores, term_weights, threshold=thr)
        if scores["f1"] > best["f1"]:
            best = {"threshold": thr, **scores}
    return best
