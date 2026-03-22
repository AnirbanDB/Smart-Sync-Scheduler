"""
schema_mapper.py

Rule-based column name mapping.

Reads known column variations from config/schema_mapping.json and attempts
to match each incoming column against the standard schema:

    store_id, product, quantity, timestamp, update_id

Matching is case-insensitive and also normalises common separators
(spaces, hyphens) to underscores before comparison.
"""

import json
import os
from typing import Dict, List, Tuple

import pandas as pd

from core.logger import get_logger

STANDARD_FIELDS = ["store_id", "product", "quantity", "timestamp", "update_id"]
REQUIRED_FIELDS = {"store_id", "product", "quantity", "timestamp"}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_mapping_rules(path: str = "config/schema_mapping.json") -> Dict[str, List[str]]:
    """Load the rule dictionary from schema_mapping.json."""
    abs_path = _resolve(path)
    try:
        with open(abs_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Rule-based mapper
# ---------------------------------------------------------------------------


def rule_based_map(
    columns: List[str],
    rules: Dict[str, List[str]],
    logger=None,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Map a list of column names to standard fields using the rules dict.

    Returns
    -------
    mapping        : {original_column: standard_field}
    unmapped_cols  : columns that could not be matched
    """
    if logger is None:
        logger = get_logger()

    mapping: Dict[str, str] = {}
    mapped_standards: set = set()

    for col in columns:
        if col in mapping:
            continue

        col_normalised = _normalise(col)

        for standard_field, variations in rules.items():
            if standard_field in mapped_standards:
                continue

            # Check: normalised column == standard field name
            if col_normalised == standard_field:
                mapping[col] = standard_field
                mapped_standards.add(standard_field)
                break

            # Check: normalised column matches any known variation
            normalised_variations = [_normalise(v) for v in variations]
            if col_normalised in normalised_variations:
                mapping[col] = standard_field
                mapped_standards.add(standard_field)
                break

    unmapped = [c for c in columns if c not in mapping]
    logger.debug(f"Rule-based mapping result: {mapping}. Unmapped: {unmapped}")
    return mapping, unmapped


def apply_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Rename DataFrame columns according to the mapping and keep only
    columns that resolve to a standard field name.
    """
    df = df.rename(columns=mapping)
    keep = [c for c in df.columns if c in STANDARD_FIELDS]
    return df[keep].copy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")


def _resolve(path: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, path)