"""
data_loader.py

Unified file loader that reads CSV, XLSX, and JSON files into a pandas
DataFrame. Returns both the DataFrame and a metadata dictionary.

Supported formats:
    .csv   - comma-separated values
    .xlsx  - Excel workbook (requires openpyxl)
    .xls   - legacy Excel (requires xlrd)
    .json  - JSON array or object with a list under a known key
"""

import json
import os
from typing import Any, Dict, Tuple

import pandas as pd

from core.logger import get_logger

_KNOWN_JSON_KEYS = ("records", "data", "updates", "items", "rows", "stock_updates")


def load_file(
    file_obj: Any,
    file_name: str,
    logger=None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load a file into a raw DataFrame.

    Parameters
    ----------
    file_obj : file-like object (from st.file_uploader or open())
    file_name : original file name, used to detect format

    Returns
    -------
    (DataFrame, file_info dict)

    Raises
    ------
    ValueError  on unsupported format or unparseable content
    """
    if logger is None:
        logger = get_logger()

    ext = os.path.splitext(file_name)[1].strip().lower()
    file_info: Dict[str, Any] = {
        "name": file_name,
        "format": ext,
        "rows": 0,
        "columns": [],
    }

    try:
        if ext == ".csv":
            df = _load_csv(file_obj)
        elif ext in (".xlsx", ".xls"):
            df = _load_excel(file_obj)
        elif ext == ".json":
            df = _load_json(file_obj)
        else:
            raise ValueError(
                f"Unsupported file format '{ext}'. Accepted: .csv, .xlsx, .xls, .json"
            )

        # Normalise column names: strip whitespace
        df.columns = [str(c).strip() for c in df.columns]

        file_info["rows"] = len(df)
        file_info["columns"] = list(df.columns)

        logger.info(
            f"Loaded '{file_name}': {len(df)} rows, columns={list(df.columns)}"
        )
        return df, file_info

    except Exception as exc:
        logger.error(f"Failed to load '{file_name}': {exc}")
        raise


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------


def _load_csv(file_obj: Any) -> pd.DataFrame:
    # Try UTF-8 first, then latin-1 as fallback
    try:
        return pd.read_csv(file_obj, encoding="utf-8")
    except UnicodeDecodeError:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        return pd.read_csv(file_obj, encoding="latin-1")


def _load_excel(file_obj: Any) -> pd.DataFrame:
    return pd.read_excel(file_obj, engine="openpyxl")


def _load_json(file_obj: Any) -> pd.DataFrame:
    raw = file_obj.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    data = json.loads(raw)

    if isinstance(data, list):
        return pd.DataFrame(data)

    if isinstance(data, dict):
        for key in _KNOWN_JSON_KEYS:
            if key in data and isinstance(data[key], list):
                return pd.DataFrame(data[key])
        # No recognised key found - treat the dict as a single record
        return pd.DataFrame([data])

    raise ValueError(
        "JSON must be an array or an object containing a list under one of: "
        + ", ".join(_KNOWN_JSON_KEYS)
    )