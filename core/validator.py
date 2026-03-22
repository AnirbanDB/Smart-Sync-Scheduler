"""
validator.py

Data validation and normalisation pipeline.

Responsibilities:
  1. Verify all required standard columns are present.
  2. Convert data types safely (quantity to int, timestamp to UTC).
  3. Normalize timestamps to UTC regardless of source timezone.
  4. Drop or flag rows that cannot be repaired.
  5. Auto-generate update_id for rows that lack one.

Timezone handling (requirement 17):
  - All timestamps are converted to timezone-aware UTC Timestamps.
  - Naive timestamps are localised to settings["default_timezone"] first.
  - Numeric Unix timestamps (seconds or milliseconds) are also supported.
  - Rows whose timestamp cannot be parsed at all are dropped and logged.
  - No naive datetime objects are stored after this stage.
"""

import uuid
from typing import List, Tuple

import pandas as pd

from core.logger import get_logger

REQUIRED_COLUMNS = {"store_id", "product", "quantity", "timestamp"}
OPTIONAL_COLUMNS = {"update_id"}
ALL_STANDARD_COLUMNS = REQUIRED_COLUMNS | OPTIONAL_COLUMNS


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def validate_and_normalize(
    df: pd.DataFrame,
    default_timezone: str = "UTC",
    logger=None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate and normalise a DataFrame that already has standard column names.

    Returns
    -------
    (clean_df, warnings)
    clean_df  : rows that passed all checks, with corrected types
    warnings  : human-readable list of issues found (empty = no problems)
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
            "Check schema_mapping.json or upload a file with matching column names."
        )

    df = df.copy()

    # 1. Auto-generate update_id for rows that lack one
    if "update_id" not in df.columns:
        df["update_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
        warnings.append(
            "Column 'update_id' was absent. Unique IDs were generated automatically."
        )
    else:
        # Fill empty / NaN update_ids with generated values
        df["update_id"] = df["update_id"].astype(str).str.strip()
        needs_id = (df["update_id"] == "") | (df["update_id"].str.lower() == "nan")
        if needs_id.any():
            df.loc[needs_id, "update_id"] = [
                str(uuid.uuid4()) for _ in range(needs_id.sum())
            ]
            warnings.append(
                f"{needs_id.sum()} row(s) had empty update_id; values were generated."
            )

    # 2. Validate store_id and product
    for col in ("store_id", "product"):
        df[col] = df[col].astype(str).str.strip()
        bad = (df[col] == "") | (df[col].str.lower() == "nan")
        if bad.any():
            warnings.append(
                f"Dropping {bad.sum()} row(s) where '{col}' is empty or null."
            )
            logger.warning(f"Dropping {bad.sum()} rows: empty '{col}'.")
        df = df[~bad]

    # 3. Validate quantity
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
            f"{negative.sum()} row(s) have negative quantity (kept - may represent returns)."
        )

    df["quantity"] = df["quantity"].astype(int)

    # 4. Normalise timestamps to UTC
    df, ts_warnings = _normalize_timestamps(df, default_timezone, logger)
    warnings.extend(ts_warnings)
    df = df.dropna(subset=["timestamp"])

    # 5. Keep only standard columns, in defined order
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

    Strategy per row:
      1. If the value is already a tz-aware Timestamp, convert to UTC.
      2. If it is a tz-naive Timestamp or string, localise to default_tz then UTC.
      3. If it looks like a number, treat as Unix epoch (auto-detect s vs ms).
      4. If none of the above work, set to NaT (the row will be dropped).
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
        logger.warning(f"{n_failed} rows will be dropped due to invalid timestamps.")

    df["timestamp"] = result
    return df, warnings


def _parse_single_timestamp(value, default_tz: str):
    """
    Parse one timestamp value to a UTC-aware pandas Timestamp.
    Returns pd.NaT on failure.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT

    # Already a Timestamp
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
            num = num / 1000.0  # milliseconds
        return pd.Timestamp(num, unit="s", tz="UTC")
    except (ValueError, TypeError, OverflowError):
        pass

    # String / object: parse then localise
    try:
        parsed = pd.Timestamp(value)
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize(default_tz, ambiguous="NaT", nonexistent="NaT")
            if pd.isna(parsed):
                return pd.NaT
        return parsed.tz_convert("UTC")
    except Exception:
        return pd.NaT