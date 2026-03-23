"""
validator.py

Data validation and normalisation pipeline.

Responsibilities:
  1. Verify all required standard columns are present.
     Required: store_id, product, quantity
     Optional: timestamp, update_id

  2. Convert data types safely.

  3. If timestamp is present, normalise it to UTC regardless of source
     timezone. If timestamp is absent, the column is simply omitted —
     the server will stamp received_at at enqueue time (see scheduler.py).
     This separates the branch's internal recording time from the server's
     receipt time, which is the architecturally correct design.

  4. Auto-generate update_id for rows that lack one.

  5. Drop rows that cannot be repaired and report them with clear messages.

Timezone handling:
  - Timezone-aware strings (+05:30, Z, etc.) are converted directly to UTC.
  - Naive strings are localised to default_timezone then converted to UTC.
  - Unix epoch values (seconds or milliseconds) are supported.
  - Rows with completely unparseable timestamps are dropped and logged.
  - No naive datetime objects survive past this stage.
"""

import uuid
from typing import List, Tuple

import pandas as pd

from core.logger import get_logger

# timestamp is now optional — the server stamps received_at at enqueue time
REQUIRED_COLUMNS = {"store_id", "product", "quantity"}
OPTIONAL_COLUMNS = {"timestamp", "update_id"}
ALL_STANDARD_COLUMNS = REQUIRED_COLUMNS | OPTIONAL_COLUMNS


def validate_and_normalize(
    df: pd.DataFrame,
    default_timezone: str = "UTC",
    logger=None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate and normalise a DataFrame that already has standard column names.

    Parameters
    ----------
    df               : DataFrame with columns mapped to the standard schema.
    default_timezone : Timezone assumed for naive timestamps.
    logger           : Optional logger instance.

    Returns
    -------
    (clean_df, warnings)
    clean_df : Rows that passed all checks, with corrected types.
    warnings : Human-readable list of issues found. Empty means no problems.
    """
    if logger is None:
        logger = get_logger()

    warnings: List[str] = []
    original_len = len(df)

    # Guard: required columns must be present
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Required columns missing after schema mapping: {sorted(missing)}. "
            "Each upload must contain at minimum: store_id, product, quantity."
        )

    df = df.copy()

    # ------------------------------------------------------------------
    # update_id: auto-generate if absent or empty
    # ------------------------------------------------------------------
    if "update_id" not in df.columns:
        df["update_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
        warnings.append(
            "Column 'update_id' was absent. Unique IDs were generated automatically."
        )
    else:
        df["update_id"] = df["update_id"].astype(str).str.strip()
        needs_id = (df["update_id"] == "") | (df["update_id"].str.lower() == "nan")
        if needs_id.any():
            df.loc[needs_id, "update_id"] = [
                str(uuid.uuid4()) for _ in range(needs_id.sum())
            ]
            warnings.append(
                f"{needs_id.sum()} row(s) had empty update_id — values generated."
            )

    # ------------------------------------------------------------------
    # store_id and product: must be non-empty strings
    # ------------------------------------------------------------------
    for col in ("store_id", "product"):
        df[col] = df[col].astype(str).str.strip()
        bad = (df[col] == "") | (df[col].str.lower() == "nan")
        if bad.any():
            warnings.append(
                f"Dropping {bad.sum()} row(s) where '{col}' is empty or null."
            )
            logger.warning(f"Dropping {bad.sum()} rows: empty '{col}'.")
        df = df[~bad]

    # ------------------------------------------------------------------
    # quantity: must be a non-negative integer
    # ------------------------------------------------------------------
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    bad_qty = df["quantity"].isna()
    if bad_qty.any():
        warnings.append(
            f"Dropping {bad_qty.sum()} row(s) with non-numeric quantity."
        )
        logger.warning(f"Dropping {bad_qty.sum()} rows: non-numeric quantity.")
    df = df.dropna(subset=["quantity"])

    negative = df["quantity"] < 0
    if negative.any():
        warnings.append(
            f"{negative.sum()} row(s) have negative quantity "
            "(kept — may represent returns or adjustments)."
        )

    df["quantity"] = df["quantity"].astype(int)

    # ------------------------------------------------------------------
    # timestamp: optional. If present, normalise to UTC.
    # If absent, the column is simply not included — scheduler.py will
    # stamp received_at when the record enters the queue.
    # ------------------------------------------------------------------
    if "timestamp" in df.columns:
        df, ts_warnings = _normalize_timestamps(df, default_timezone, logger)
        warnings.extend(ts_warnings)
        # Rows with completely unparseable timestamps are dropped
        df = df.dropna(subset=["timestamp"])
    else:
        warnings.append(
            "Column 'timestamp' was not provided. "
            "The server will record the upload time as the effective timestamp."
        )

    # ------------------------------------------------------------------
    # Keep only standard columns in a defined order
    # ------------------------------------------------------------------
    col_order = [c for c in ALL_STANDARD_COLUMNS if c in df.columns]
    df = df[col_order].reset_index(drop=True)

    dropped = original_len - len(df)
    if dropped:
        warnings.append(
            f"Total: {dropped} of {original_len} row(s) dropped during validation."
        )

    logger.info(
        f"Validation complete: {len(df)} valid rows, {dropped} dropped."
    )
    return df, warnings


# ---------------------------------------------------------------------------
# Timestamp normalisation
# ---------------------------------------------------------------------------

def _normalize_timestamps(
    df: pd.DataFrame,
    default_tz: str,
    logger,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert every value in df["timestamp"] to a UTC-aware pandas Timestamp.

    Strategy per value:
      1. Already tz-aware  -> convert to UTC directly.
      2. Tz-naive or string -> localise to default_tz then convert to UTC.
      3. Numeric            -> treat as Unix epoch (auto-detect s vs ms).
      4. Unparseable        -> set NaT (row will be dropped).
    """
    warnings: List[str] = []

    result = df["timestamp"].apply(
        lambda v: _parse_single_timestamp(v, default_tz)
    )

    n_failed = result.isna().sum()
    if n_failed:
        warnings.append(
            f"{n_failed} row(s) had unparseable timestamps and will be dropped."
        )
        logger.warning(f"{n_failed} rows will be dropped: invalid timestamps.")

    df["timestamp"] = result
    return df, warnings


def _parse_single_timestamp(value, default_tz: str):
    """Return a UTC-aware pandas Timestamp, or pd.NaT on failure."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT

    if isinstance(value, pd.Timestamp):
        try:
            if value.tzinfo is None:
                value = value.tz_localize(default_tz)
            return value.tz_convert("UTC")
        except Exception:
            return pd.NaT

    # Numeric: Unix epoch
    try:
        num = float(value)
        if num > 1e12:
            num /= 1000.0
        return pd.Timestamp(num, unit="s", tz="UTC")
    except (ValueError, TypeError, OverflowError):
        pass

    # String or object
    try:
        parsed = pd.Timestamp(value)
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize(default_tz, ambiguous="NaT", nonexistent="NaT")
            if pd.isna(parsed):
                return pd.NaT
        return parsed.tz_convert("UTC")
    except Exception:
        return pd.NaT