"""
Lightweight Gene Ontology graph utilities for propagation and validation.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Set, Tuple


@dataclass
class GOTermGraph:
    """Minimal GO graph representation using parent adjacency."""

    parents: Dict[str, Set[str]]

    @classmethod
    def from_edges(cls, edges: Iterable[Tuple[str, str]]) -> "GOTermGraph":
        """
        Build a GO graph from (child, parent) edges.

        Args:
            edges: Iterable of (child, parent) pairs.
        """
        parent_map: Dict[str, Set[str]] = defaultdict(set)
        for child, parent in edges:
            parent_map[child].add(parent)
            parent_map.setdefault(parent, set())
        return cls(parents=dict(parent_map))

    @classmethod
    def from_tsv(cls, path: Path, delimiter: str = "\t") -> "GOTermGraph":
        """
        Build a graph from a 2-column TSV/CSV file: child<TAB>parent.
        """
        edges: List[Tuple[str, str]] = []
        with Path(path).open("r") as f:
            for line in f:
                parts = line.strip().split(delimiter)
                if len(parts) >= 2:
                    edges.append((parts[0], parts[1]))
        return cls.from_edges(edges)

    @property
    def terms(self) -> Set[str]:
        """Return all terms present in the graph."""
        return set(self.parents.keys())

    def ancestors(self, term: str) -> Set[str]:
        """Return all ancestors of a term (parents, grandparents, ...)."""
        visited: Set[str] = set()
        queue: deque[str] = deque(self.parents.get(term, set()))
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            queue.extend(self.parents.get(node, []))
        return visited

    def propagate_scores(self, scores: Mapping[str, float]) -> Dict[str, float]:
        """
        Propagate term scores upward so that every parent receives the max of its children.

        Args:
            scores: Mapping of term -> score.

        Returns:
            New mapping with propagated parent scores.
        """
        propagated: Dict[str, float] = dict(scores)
        for term, score in scores.items():
            for parent in self.ancestors(term):
                propagated[parent] = max(propagated.get(parent, 0.0), score)
        return propagated

    def invalid_terms(self, terms: Iterable[str]) -> Set[str]:
        """Identify terms not present in the graph."""
        term_set = set(terms)
        return term_set - self.terms


def cap_terms(term_scores: Mapping[str, float], max_terms: int) -> Dict[str, float]:
    """
    Cap a term-score mapping to the top-N terms by score (descending).
    """
    sorted_terms = sorted(term_scores.items(), key=lambda kv: kv[1], reverse=True)
    trimmed = dict(sorted_terms[:max_terms])
    return trimmed
