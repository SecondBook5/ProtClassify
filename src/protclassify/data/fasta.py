"""Lightweight FASTA parsing utilities for CAFA6 datasets.

The CAFA organizers distribute target sequences as FASTA files.  We keep a
minimal dependency surface (no Biopython required) to make it easy to parse
those files inside training and inference pipelines.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd


@dataclass
class FastaRecord:
    """Simple container for a FASTA entry."""

    identifier: str
    sequence: str
    description: str | None = None


def parse_fasta(path: str | Path) -> Iterator[FastaRecord]:
    """Yield :class:`FastaRecord` objects from a FASTA file.

    Parameters
    ----------
    path:
        Location of the FASTA file.
    """

    current_id: str | None = None
    current_desc: str | None = None
    current_seq: list[str] = []

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    yield FastaRecord(
                        identifier=current_id,
                        sequence="".join(current_seq),
                        description=current_desc,
                    )
                header = line[1:]
                parts = header.split(maxsplit=1)
                current_id = parts[0]
                current_desc = parts[1] if len(parts) > 1 else None
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        yield FastaRecord(
            identifier=current_id,
            sequence="".join(current_seq),
            description=current_desc,
        )


def load_fasta_as_dataframe(path: str | Path) -> pd.DataFrame:
    """Parse a FASTA file into a tidy :class:`pandas.DataFrame`.

    Returns a DataFrame with columns ``["protein_id", "sequence", "description"]``
    to make it easy to merge with label tables or feature matrices.
    """

    records = list(parse_fasta(path))
    return pd.DataFrame(
        {
            "protein_id": [r.identifier for r in records],
            "sequence": [r.sequence for r in records],
            "description": [r.description for r in records],
        }
    )


def write_fasta(records: Sequence[FastaRecord], output_path: str | Path) -> Path:
    """Serialize FASTA records to disk.

    Useful for exporting filtered subsets (e.g., a CAFA6 validation slice) for
    external scoring tools.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            description = f" {record.description}" if record.description else ""
            handle.write(f">{record.identifier}{description}\n")
            handle.write("\n".join([record.sequence[i : i + 80] for i in range(0, len(record.sequence), 80)]))
            handle.write("\n")

    return output_path


__all__ = [
    "FastaRecord",
    "parse_fasta",
    "load_fasta_as_dataframe",
    "write_fasta",
]
