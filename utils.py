"""Read/write tabular datasets for the parsing pipeline (pandas, TSV)."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

# PARSE.md: minimum required columns in the input CSV
REQUIRED_COLUMNS = ("category", "content")


def _abspath(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(_abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_documents_csv(path: str, *, sep: str = "\t", encoding: str = "utf-8") -> pd.DataFrame:
    """
    Load CSV/TSV. Default separator is tab.
    Column names are normalized (strip). ``category`` and ``content`` are required.
    """
    path = _abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path, sep=sep, encoding=encoding, dtype=str, keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in {path}. "
            f"Found: {list(df.columns)}. See PARSE.md."
        )

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def write_documents_csv(
    df: pd.DataFrame,
    path: str,
    *,
    sep: str = "\t",
    encoding: str = "utf-8",
) -> str:
    """Write dataframe as TSV (default: tab, UTF-8)."""
    path = _abspath(path)
    _ensure_parent_dir(path)
    df.to_csv(path, sep=sep, index=False, encoding=encoding, lineterminator="\n")
    return path


def csv_cell_has_value(x: Any) -> bool:
    """True if a cell is non-empty (after ``read_documents_csv`` normalization)."""
    if x is None:
        return False
    s = str(x).strip()
    if not s:
        return False
    low = s.lower()
    if low in ("nan", "none", "<na>"):
        return False
    return True

