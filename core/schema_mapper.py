"""
schema_mapper.py

Rule-based column name mapping with three progressively broader matching
levels, so columns survive mapping regardless of surrounding words or
minor phrasing variations.

Matching is attempted in order per column:

  Level 1 — Exact normalised match
    The normalised column name matches a standard field name or a known
    synonym from schema_mapping.json exactly.
    Example: "qty" -> "quantity",  "shop" -> "store_id"

  Level 2 — High-specificity token match
    The column name (split into tokens) contains a domain-specific token
    that unambiguously signals one field.  Only tokens that would not
    appear as contextual words in another field's columns are used here.
    Example: "no. of pdts that are available"
             token "pdts" is quantity-specific -> "quantity"
    Example: "Name of the medicine to be used by persons"
             token "medicine" is product-specific -> "product"

  Level 3 — Semantic keyword containment match
    The lowercased column string contains a broader pharmacy-domain
    keyword, evaluated in strict priority order:
        update_id > timestamp > store_id > product > quantity
    Priority prevents "stock" inside "date of stock capture" from
    overriding the unambiguous "date" timestamp signal.

The LLM only receives columns that survive all three levels without a
match — in practice this should be rare for standard pharmacy data.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from core.logger import get_logger

STANDARD_FIELDS = ["store_id", "product", "quantity", "timestamp", "update_id"]
REQUIRED_FIELDS = {"store_id", "product", "quantity"}


# ---------------------------------------------------------------------------
# Level 2 — High-specificity tokens
#
# Only tokens that are unambiguously pharmacy-domain AND would not
# routinely appear as contextual words in another field's column name.
#
# Deliberately excluded from this table:
#   "branch", "outlet", "location" — appear in medicine column names
#   "stock" — appears in "date of stock capture" (timestamp)
#   "units" — appears in "units of active ingredient" (product)
#   "count" — too generic
# ---------------------------------------------------------------------------

_L2_SPECIFIC_TOKENS: Dict[str, set] = {
    "update_id": {
        "txnid", "transactionid", "recordid", "rowid",
        "seqno", "batchid", "syncid", "entryno",
    },
    "timestamp": {
        "timestamp", "datetime",
    },
    "store_id": set(),   # no single token is specific enough for store_id
    "product": {
        "medicine", "medicines", "medication", "medications",
        "drug", "drugs", "pharmaceutical", "pharmaceuticals",
        "tablet", "tablets", "capsule", "capsules",
        "formulation", "formulations", "preparation", "preparations",
        "ingredient", "ingredients",
        "dawa",   # Swahili/Hindi: medicine
        "ubat",   # Malay: medicine
    },
    "quantity": {
        "qty", "pdts", "pdt",
        "nos",    # "no. of ..." patterns
        "matra",  # Hindi: quantity/dose
        "stok",   # Malay: stock
    },
}

# Pre-tokenise: split on any non-alphanumeric character
_TOKEN_RE = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# Level 3 — Semantic keyword signals
#
# Broader signals used when Level 2 finds nothing.
# Evaluated in strict priority order — first match per column wins.
# Priority: update_id > timestamp > store_id > product > quantity
# ---------------------------------------------------------------------------

_L3_SIGNALS: List[Tuple[str, List[str]]] = [
    ("update_id", [
        "reference", "ref", "serial", "sequence", "revision",
        "version", "batch",
    ]),
    ("timestamp", [
        "date", "time", "recorded", "updated", "created", "modified",
        "logged", "captured", "reported", "synced", "samay", "masa",
    ]),
    ("store_id", [
        "branch", "outlet", "pharmacy", "dispensary", "chemist",
        "shop", "facility", "site", "location", "depot",
        "dawakhana", "kedai",
    ]),
    ("product", [
        "item", "product", "medicine", "medication", "drug",
        "pharmaceutical", "tablet", "capsule", "formulation",
        "preparation", "ingredient", "sku", "remedy",
    ]),
    ("quantity", [
        "qty", "quantity", "stock", "inventory", "units",
        "count", "amount", "balance", "packs", "pieces",
        "available", "avail", "onhand", "closing", "opening",
        "pdts", "nos", "matra", "stok",
    ]),
]

_L3_PATTERNS: List[Tuple[str, List[re.Pattern]]] = [
    (
        field,
        [
            re.compile(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])")
            for kw in keywords
        ],
    )
    for field, keywords in _L3_SIGNALS
]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_mapping_rules(path: str = "config/schema_mapping.json") -> Dict[str, List[str]]:
    """Load the synonym dictionary from schema_mapping.json."""
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
# Public mapper
# ---------------------------------------------------------------------------


def rule_based_map(
    columns: List[str],
    rules: Dict[str, List[str]],
    logger=None,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Map a list of column names to standard fields.

    Applies Level 1, 2, then 3 in order.  Columns that survive all three
    levels without a match are returned in unmapped_cols for LLM inference.

    Returns
    -------
    mapping       : {original_column: standard_field}
    unmapped_cols : columns that could not be matched by any rule level
    """
    if logger is None:
        logger = get_logger()

    mapping: Dict[str, str] = {}
    mapped_standards: set = set()

    for col in columns:
        if col in mapping:
            continue

        col_normalised = _normalise(col)
        matched = (
            _level1_match(col, col_normalised, rules, mapped_standards)
            or _level2_match(col, mapped_standards)
            or _level3_match(col, mapped_standards)
        )

        if matched is not None:
            mapping[col] = matched
            mapped_standards.add(matched)

    unmapped = [c for c in columns if c not in mapping]
    logger.debug(
        f"Rule-based mapping: {mapping}.  "
        f"Unmapped (sent to LLM): {unmapped}"
    )
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
# Matching level implementations (also used by tests and llm_mapper
# validation for normalised key lookup)
# ---------------------------------------------------------------------------


def _level1_match(
    col: str,
    col_normalised: str,
    rules: Dict[str, List[str]],
    already_mapped: set,
) -> Optional[str]:
    """
    Exact match against standard field names and known synonyms.
    """
    for standard_field, variations in rules.items():
        if standard_field in already_mapped:
            continue
        if col_normalised == standard_field:
            return standard_field
        if col in variations:
            return standard_field
        if col_normalised in (_normalise(v) for v in variations):
            return standard_field
    return None


def _level2_match(col: str, already_mapped: set) -> Optional[str]:
    """
    High-specificity token match.

    Splits the column name into lowercase alphanumeric tokens and checks
    each token against _L2_SPECIFIC_TOKENS.  Only tokens that are
    unambiguously domain-specific (not common English context words) are
    included in the table.
    """
    tokens = set(_TOKEN_RE.split(col.strip().lower()))
    tokens.discard("")

    for standard_field, specific_set in _L2_SPECIFIC_TOKENS.items():
        if standard_field in already_mapped:
            continue
        if tokens & specific_set:
            return standard_field
    return None


def _level3_match(col: str, already_mapped: set) -> Optional[str]:
    """
    Semantic keyword containment match using broader signals.

    Evaluated in strict priority order so an unambiguous signal like
    "date" or "time" beats a weaker one like "stock" when both appear
    in the same column name.
    """
    col_lower = col.strip().lower()
    for standard_field, patterns in _L3_PATTERNS:
        if standard_field in already_mapped:
            continue
        for pattern in patterns:
            if pattern.search(col_lower):
                return standard_field
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
    )


def _resolve(path: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, path)