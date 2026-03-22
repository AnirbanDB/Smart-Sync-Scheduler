"""
llm_mapper.py

LLM-based semantic schema mapping.

When rule-based mapping cannot resolve all required columns, this module
calls a configurable LLM endpoint (Ollama, OpenAI-compatible, etc.) to
infer the correct mapping.

Safety guarantees enforced here (requirement 22):
  - All LLM output is validated before use.
  - Hallucinated or out-of-schema field names are rejected.
  - Confidence below threshold triggers fallback.

Caching (requirement 23):
  - Mappings are cached in-memory and persisted to a JSON file, keyed by
    a hash of the sorted column names. Repeated uploads of the same schema
    never make a second API call.

Rate limiting (requirement 23):
  - A sliding-window counter prevents more than rate_limit_per_minute calls
    in any 60-second window. Excess requests fall back to rule-based mapping.

Graceful degradation (requirement 24):
  - Any failure (timeout, JSON parse error, low confidence) returns
    (None, 0.0, reason) instead of raising an exception.
"""

import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import requests

from core.logger import get_logger
from core.schema_mapper import STANDARD_FIELDS, REQUIRED_FIELDS

# ---------------------------------------------------------------------------
# Module-level state (persists across Streamlit reruns within one process)
# ---------------------------------------------------------------------------

_memory_cache: Dict[str, Dict] = {}
_request_timestamps: List[float] = []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def llm_map_columns(
    columns: List[str],
    llm_config: Dict,
    already_mapped: Dict[str, str],
    logger=None,
) -> Tuple[Optional[Dict[str, str]], float, str]:
    """
    Attempt LLM-based mapping for columns not covered by rule-based mapping.

    Parameters
    ----------
    columns        : all original column names from the uploaded file
    llm_config     : the "llm" sub-dict from settings.json
    already_mapped : mapping already produced by the rule-based step
                     (used to determine which columns still need mapping)

    Returns
    -------
    (mapping, confidence, reason)
    mapping    : {original_col: standard_field} or None on failure
    confidence : float 0.0-1.0
    reason     : short human-readable status string
    """
    if logger is None:
        logger = get_logger()

    if not llm_config.get("enabled", False):
        return None, 0.0, "LLM disabled in configuration."

    unmapped = [c for c in columns if c not in already_mapped]
    if not unmapped:
        return {}, 1.0, "All columns already covered by rule-based mapping."

    # Cache lookup
    sig = _column_signature(unmapped)
    cache_file = llm_config.get("cache_file", "data/llm_cache.json")

    global _memory_cache
    if not _memory_cache:
        _memory_cache = _load_cache(cache_file)

    if sig in _memory_cache:
        entry = _memory_cache[sig]
        logger.info(f"LLM cache hit (sig={sig[:8]}).")
        return entry["mapping"], entry["confidence"], "cached"

    # Rate limit check
    rate_limit = int(llm_config.get("rate_limit_per_minute", 10))
    if not _check_rate_limit(rate_limit):
        logger.warning("LLM rate limit exceeded. Using rule-based fallback.")
        return None, 0.0, "Rate limit exceeded. Falling back to rule-based mapping."

    # Build prompt
    prompt = _build_prompt(unmapped)

    # Call API (with retry)
    endpoint = llm_config.get("endpoint", "http://localhost:11434/api/generate")
    model = llm_config.get("model", "qwen2.5:7b")
    timeout = int(llm_config.get("timeout", 15))
    api_retries = int(llm_config.get("api_retries", 2))

    api_key = llm_config.get("api_key", "")
    provider = llm_config.get("provider", "ollama")

    raw_response = _call_api(
        endpoint=endpoint,
        model=model,
        prompt=prompt,
        timeout=timeout,
        retries=api_retries,
        api_key=api_key,
        provider=provider,
        logger=logger,
    )

    if raw_response is None:
        return None, 0.0, "LLM API did not return a response."

    # Parse JSON from response
    parsed = _extract_json(raw_response)
    if parsed is None:
        logger.warning("LLM response could not be parsed as JSON.")
        return None, 0.0, "LLM response was not valid JSON."

    confidence = float(parsed.get("confidence", 0.5))
    threshold = float(llm_config.get("confidence_threshold", 0.7))

    # Strip the confidence key from the column mapping
    col_mapping = {k: v for k, v in parsed.items() if k != "confidence"}

    # Validate the mapping (requirement 22)
    valid, reason = _validate_mapping(col_mapping, columns)
    if not valid:
        logger.warning(f"LLM mapping validation failed: {reason}")
        return None, confidence, f"LLM mapping rejected: {reason}"

    if confidence < threshold:
        logger.warning(
            f"LLM confidence {confidence:.2f} is below threshold {threshold:.2f}."
        )
        return None, confidence, (
            f"LLM confidence {confidence:.2f} below threshold {threshold:.2f}. "
            "Falling back to rule-based mapping."
        )

    # Cache and return
    _memory_cache[sig] = {"mapping": col_mapping, "confidence": confidence}
    _save_cache(_memory_cache, cache_file)
    logger.info(f"LLM mapping accepted (confidence={confidence:.2f}).")
    return col_mapping, confidence, "ok"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_prompt(unmapped_columns: List[str]) -> str:
    return (
        "You are a data schema mapping assistant for a pharmacy stock system.\n\n"
        f"Standard schema fields: {', '.join(STANDARD_FIELDS)}\n"
        f"Required fields: {', '.join(sorted(REQUIRED_FIELDS))}\n\n"
        f"Column names to map: {', '.join(unmapped_columns)}\n\n"
        "Rules:\n"
        "- Return ONLY a valid JSON object. No explanation, no markdown.\n"
        f"- Keys must be the original column names from the list above.\n"
        f"- Values must be one of: {', '.join(STANDARD_FIELDS)}.\n"
        "- Include a 'confidence' key with a float from 0.0 to 1.0.\n"
        "- Only include mappings you are confident about.\n"
        "- Do not invent column names not in the list.\n\n"
        'Example: {"branch": "store_id", "med": "product", "qty": "quantity", '
        '"ts": "timestamp", "confidence": 0.92}'
    )


def _call_api(
    endpoint: str,
    model: str,
    prompt: str,
    timeout: int,
    retries: int,
    api_key: str,
    provider: str,
    logger,
) -> Optional[str]:
    """
    Call the LLM API.

    provider="openai"  -> sends a chat/completions request with Bearer auth.
    provider="ollama"  -> sends an Ollama /api/generate request (no auth).

    Any other value is treated as an OpenAI-compatible endpoint, which is
    the safest default for third-party hosted models.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(retries + 1):
        try:
            if provider == "ollama":
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,
                }
            else:
                # OpenAI and all OpenAI-compatible providers (Groq, Together, etc.)
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                }

            resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            # Extract text from whichever response shape the provider returned
            if "response" in data:
                # Ollama shape
                return data["response"]
            if "choices" in data:
                # OpenAI shape
                return data["choices"][0]["message"]["content"]
            return str(data)

        except requests.exceptions.Timeout:
            logger.warning(f"LLM API timeout (attempt {attempt + 1}/{retries + 1}).")
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            logger.warning(
                f"LLM API HTTP error {status} (attempt {attempt + 1}/{retries + 1}): {exc}"
            )
            # 401 / 403 are auth errors - no point retrying
            if status in (401, 403):
                logger.error(
                    "Authentication failed. Check that 'api_key' in settings.json is correct."
                )
                return None
        except Exception as exc:
            logger.warning(f"LLM API error (attempt {attempt + 1}/{retries + 1}): {exc}")

        if attempt < retries:
            time.sleep(1.5)

    return None


def _extract_json(raw: str) -> Optional[Dict]:
    """Extract the first JSON object from an arbitrary string response."""
    raw = raw.strip()

    # Strip markdown code fences if present
    for fence in ("```json", "```"):
        if fence in raw:
            parts = raw.split(fence)
            if len(parts) >= 3:
                raw = parts[1].strip()
                break

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return None

    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None


def _validate_mapping(mapping: Dict[str, str], original_columns: List[str]) -> Tuple[bool, str]:
    """Ensure the LLM mapping is safe to use (requirement 22)."""
    if not mapping:
        return False, "Mapping is empty."

    orig_lower = {c.lower(): c for c in original_columns}

    for col, field in mapping.items():
        if field not in STANDARD_FIELDS:
            return False, f"'{col}' mapped to unknown field '{field}'."
        if col not in original_columns and col.lower() not in orig_lower:
            return False, f"Column '{col}' does not exist in the uploaded file."

    return True, "ok"


def _column_signature(columns: List[str]) -> str:
    normalised = sorted(c.strip().lower() for c in columns)
    return hashlib.md5("|".join(normalised).encode()).hexdigest()


def _check_rate_limit(rate_limit_per_minute: int) -> bool:
    global _request_timestamps
    now = time.time()
    _request_timestamps = [t for t in _request_timestamps if now - t < 60]
    if len(_request_timestamps) >= rate_limit_per_minute:
        return False
    _request_timestamps.append(now)
    return True


def _load_cache(cache_path: str) -> Dict:
    abs_path = _resolve(cache_path)
    try:
        if os.path.isfile(abs_path):
            with open(abs_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        pass
    return {}


def _save_cache(cache: Dict, cache_path: str) -> None:
    abs_path = _resolve(cache_path)
    try:
        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2)
    except Exception:
        pass


def _resolve(path: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, path)