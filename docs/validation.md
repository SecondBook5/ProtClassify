# CAFA6 Validation & Submission Checklist

## Metrics
- Use information-accretion weighted precision/recall/F1 (Jiang et al. 2016). See `src/protclassify/metrics.py`.
- Sweep thresholds per ontology to maximize weighted F1 (`maximize_f1_over_thresholds`).
- Report bootstrap confidence intervals for MF/BP/CC separately.

## GO Graph Handling
- Load GO child → parent edges into `GOTermGraph`.
- Propagate scores upward so parents are never lower than children (`propagate_scores`).
- Cap predictions at 1,500 terms per target after propagation.

## Submission Validation
1. Build raw predictions per target: `{target: {go_term: score}}`.
2. Provide allowed GO terms from the release used by CAFA6.
3. Run `prepare_predictions` to filter invalid terms, propagate, and cap.
4. Convert to CAFA TSV via `write_submission_tsv` (no header, tab-separated).

## Example (pseudocode)
```
graph = GOTermGraph.from_edges(go_edges)
cleaned, report = prepare_predictions(raw_preds, graph, allowed_terms=go_terms, max_terms=1500)
write_submission_tsv(cleaned, Path("artifacts/submissions/cafa6_attempt1.tsv"))
```

Keep the `report` output with your experiment log to trace any invalid GO IDs or dropped terms.
