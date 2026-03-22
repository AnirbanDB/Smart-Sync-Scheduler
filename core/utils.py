"""
utils.py

Shared utility functions:
  - settings.json loader
  - DataFrame formatters for Streamlit display tables
  - Timestamp serialisation helper (UTC Timestamp -> ISO string)
"""

import json
import os
from typing import Any, Dict, List

import pandas as pd


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "max_sync_per_window": 5,
    "conflict_policy": "multi_level",
    "retry_limit": 3,
    "scheduling_strategy": "fifo",
    "default_timezone": "UTC",
    "app_title": "Pharmacy Smart Sync Scheduler",
    "app_subtitle": "Queue-based stock synchronisation across pharmacy branches",
    "idempotency_store_path": "data/processed_ids.json",
    "log_file": "data/sync.log",
    "llm": {
        "enabled": False,
        "endpoint": "http://localhost:11434/api/generate",
        "model": "qwen2.5:7b",
        "timeout": 15,
        "confidence_threshold": 0.7,
        "rate_limit_per_minute": 10,
        "api_retries": 2,
        "cache_file": "data/llm_cache.json",
    },
}


def load_settings(path: str = "config/settings.json") -> Dict[str, Any]:
    """
    Load settings from JSON, falling back to defaults for any missing key.
    If the file is unreadable the full default set is returned.
    """
    settings = _deep_copy(_DEFAULTS)
    abs_path = _resolve(path)
    if not os.path.isfile(abs_path):
        return settings
    try:
        with open(abs_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return settings
        for k, v in data.items():
            settings[k] = v
    except (json.JSONDecodeError, OSError):
        pass
    return settings


# ---------------------------------------------------------------------------
# DataFrame formatters
# ---------------------------------------------------------------------------


def records_to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format raw upload records for display."""
    if not records:
        return pd.DataFrame(columns=["store_id", "product", "quantity", "timestamp", "update_id"])
    df = pd.DataFrame(records)
    cols = [c for c in ["store_id", "product", "quantity", "timestamp", "update_id"] if c in df.columns]
    df = df[cols].copy()
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].apply(_ts_to_str)
    return df.reset_index(drop=True)


def queue_to_df(pending: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format pending queue records for display, with a position column."""
    if not pending:
        return pd.DataFrame(columns=["position", "store_id", "product", "quantity", "timestamp", "update_id"])
    df = pd.DataFrame(pending)
    cols = [c for c in ["store_id", "product", "quantity", "timestamp", "update_id"] if c in df.columns]
    df = df[cols].copy()
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].apply(_ts_to_str)
    df.insert(0, "position", range(1, len(df) + 1))
    return df.reset_index(drop=True)


def processed_to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format the processed-records audit log for display."""
    if not records:
        return pd.DataFrame(
            columns=["sync_cycle", "store_id", "product", "quantity", "timestamp", "update_id", "sync_status"]
        )
    df = pd.DataFrame(records)
    cols = [
        c for c in
        ["sync_cycle", "store_id", "product", "quantity", "timestamp", "update_id", "sync_status"]
        if c in df.columns
    ]
    df = df[cols].copy()
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].apply(_ts_to_str)
    return df.reset_index(drop=True)


def database_to_df(db_records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format central database records for display."""
    if not db_records:
        return pd.DataFrame(
            columns=["store_id", "product", "quantity", "last_updated", "synced_at", "update_id"]
        )
    df = pd.DataFrame(db_records)
    cols = [
        c for c in
        ["store_id", "product", "quantity", "last_updated", "synced_at", "update_id", "sync_cycle"]
        if c in df.columns
    ]
    df = df[cols].copy()
    if "last_updated" in df.columns:
        df["last_updated"] = df["last_updated"].apply(_ts_to_str)
    return df.sort_values(["store_id", "product"]).reset_index(drop=True)


def dead_letter_to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Format dead letter queue records for display.

    Columns shown:
        store_id, product, quantity, update_id, retries, failure_reason
    The retry_count column is renamed to 'retries' for clarity.
    failure_reason is truncated to 80 chars to keep the table readable.
    """
    if not records:
        return pd.DataFrame(
            columns=["store_id", "product", "quantity", "update_id", "retries", "failure_reason"]
        )
    df = pd.DataFrame(records)
    cols = [
        c for c in
        ["store_id", "product", "quantity", "update_id", "retry_count", "failure_reason"]
        if c in df.columns
    ]
    df = df[cols].copy()

    # Rename retry_count -> retries
    if "retry_count" in df.columns:
        df = df.rename(columns={"retry_count": "retries"})

    # Truncate long reasons to keep the table clean
    if "failure_reason" in df.columns:
        df["failure_reason"] = df["failure_reason"].astype(str).str.slice(0, 80)

    return df.reset_index(drop=True)


def mapping_to_df(mapping: Dict[str, str], source: Dict[str, str]) -> pd.DataFrame:
    """Display schema mapping result as a readable table."""
    rows = []
    for orig, standard in mapping.items():
        rows.append({
            "original_column": orig,
            "mapped_to": standard,
            "source": source.get(orig, "unknown"),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["original_column", "mapped_to", "source"]
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ts_to_str(ts) -> str:
    """Convert a pandas Timestamp (or anything) to a readable ISO string."""
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return ""
    try:
        if hasattr(ts, "isoformat"):
            return ts.isoformat(timespec="seconds")
        return str(ts)
    except Exception:
        return str(ts)


def _deep_copy(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_copy(i) for i in obj]
    return obj


def _resolve(path: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, path)