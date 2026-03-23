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

    # Strip the confidence key before passing to the validator
    raw_col_mapping = {k: v for k, v in parsed.items() if k != "confidence"}

    # Validate and clean — bad individual entries are discarded, not the
    # whole mapping. This prevents one hallucinated key from silently
    # discarding every correctly-mapped column.
    col_mapping, rejected = _validate_and_clean_mapping(raw_col_mapping, columns)

    if rejected:
        for reason in rejected:
            logger.warning(f"LLM mapping partial rejection:{reason}")

    if not col_mapping:
        # Every single entry was invalid — genuine failure
        logger.warning("LLM mapping rejected entirely: no valid entries survived validation.")
        first_rejection = rejected[0] if rejected else "empty mapping"
        return None, confidence, f"LLM mapping rejected: {first_rejection.strip()}"

    if confidence < threshold:
        logger.warning(
            f"LLM confidence {confidence:.2f} is below threshold {threshold:.2f}."
        )
        return None, confidence, (
            f"LLM confidence {confidence:.2f} below threshold {threshold:.2f}. "
            "Falling back to rule-based mapping."
        )

    # Only cache after successful validation so a bad response never
    # poisons the cache for future uploads of the same schema.
    _memory_cache[sig] = {"mapping": col_mapping, "confidence": confidence}
    _save_cache(_memory_cache, cache_file)

    n_clean = len(col_mapping)
    n_dropped = len(rejected)
    status = (
        "ok" if n_dropped == 0
        else f"partial: {n_clean} column(s) mapped, {n_dropped} entry(s) discarded"
    )
    logger.info(f"LLM mapping accepted (confidence={confidence:.2f}, status={status}).")
    return col_mapping, confidence, status


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Few-shot example bank
#
# Each entry is a realistic set of column names a pharmacy branch might
# actually use, paired with the correct JSON output the LLM should produce.
#
# Coverage:
#   - Short abbreviations  (qty, loc, med, ts, uid)
#   - Long descriptive names  ("Name of the medicine to be used")
#   - CamelCase / PascalCase  (drugName, StoreCode, OnHandQty)
#   - ALLCAPS database exports  (BRANCH_CODE, PRODUCT_DESC)
#   - ERP / POS system names  (SiteRef, ItemDescription, OnHandQty)
#   - Regional / transliterated names  (dawakhana, kedai, ubat)
#   - Ambiguous but inferrable columns  (ref, description, balance)
#   - Mixed files where some columns are already rule-mapped and only
#     a subset of descriptive columns arrives here for LLM inference
#
# Adding new examples:
#   Append a dict with keys "input" (list of column name strings),
#   "output" (the correct JSON mapping including "confidence"),
#   and "note" (one-line explanation for developers).
#   Keep each example to at most 6 columns so examples stay compact.
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES: List[dict] = [

    # ---- Short abbreviations ------------------------------------------------
    {
        "input":  ["loc", "med", "qty"],
        "output": {"loc": "store_id", "med": "product", "qty": "quantity",
                   "confidence": 0.97},
        "note": "loc=location/store, med=medicine, qty=quantity",
    },
    {
        "input":  ["br", "drug", "amt", "ts", "uid"],
        "output": {"br": "store_id", "drug": "product", "amt": "quantity",
                   "ts": "timestamp", "uid": "update_id", "confidence": 0.95},
        "note": "br=branch, amt=amount of stock, ts=timestamp, uid=unique id",
    },
    {
        "input":  ["ph_id", "item", "on_hand", "rec_time"],
        "output": {"ph_id": "store_id", "item": "product",
                   "on_hand": "quantity", "rec_time": "timestamp",
                   "confidence": 0.96},
        "note": "ph_id=pharmacy id, on_hand=on-hand stock count",
    },
    {
        "input":  ["sid", "prod", "cnt", "dt"],
        "output": {"sid": "store_id", "prod": "product",
                   "cnt": "quantity", "dt": "timestamp",
                   "confidence": 0.94},
        "note": "sid=store id, prod=product, cnt=count, dt=date",
    },

    # ---- Long descriptive names (real-world problem case) -------------------
    {
        "input":  ["Name of the medicine to be used", "no. of pdts"],
        "output": {"Name of the medicine to be used": "product",
                   "no. of pdts": "quantity", "confidence": 0.94},
        "note": "Full English descriptions — pdts=products=quantity",
    },
    {
        "input":  ["Pharmacy Branch Code", "Medicine Description",
                   "Available Stock Count", "Last Updated Time"],
        "output": {"Pharmacy Branch Code": "store_id",
                   "Medicine Description": "product",
                   "Available Stock Count": "quantity",
                   "Last Updated Time": "timestamp", "confidence": 0.97},
        "note": "Verbose ERP-style column headers",
    },
    {
        "input":  ["Store Location Name", "Drug / Product Name",
                   "Total Units On Hand", "Record Timestamp", "Row ID"],
        "output": {"Store Location Name": "store_id",
                   "Drug / Product Name": "product",
                   "Total Units On Hand": "quantity",
                   "Record Timestamp": "timestamp",
                   "Row ID": "update_id", "confidence": 0.96},
        "note": "Slash in column name, Row ID=update_id",
    },
    {
        "input":  ["number of units available", "medicine name",
                   "branch identifier", "date and time of report"],
        "output": {"branch identifier": "store_id",
                   "medicine name": "product",
                   "number of units available": "quantity",
                   "date and time of report": "timestamp",
                   "confidence": 0.95},
        "note": "All-lowercase full phrase columns",
    },
    {
        "input":  ["shop", "Name of the medicine to be used",
                   "no. of pdts", "timestamp", "update_id"],
        "output": {"shop": "store_id",
                   "Name of the medicine to be used": "product",
                   "no. of pdts": "quantity", "confidence": 0.96},
        "note": (
            "Mixed file — shop/timestamp/update_id already handled by "
            "rule-based pass, only the descriptive columns need LLM"
        ),
    },

    # ---- CamelCase / PascalCase (JSON exports, ORMs) ------------------------
    {
        "input":  ["drugName", "storeCode", "stockQty", "recordedAt", "txnId"],
        "output": {"storeCode": "store_id", "drugName": "product",
                   "stockQty": "quantity", "recordedAt": "timestamp",
                   "txnId": "update_id", "confidence": 0.97},
        "note": "camelCase JSON-style keys",
    },
    {
        "input":  ["MedicineName", "OutletId", "UnitsInStock", "SyncVersion"],
        "output": {"OutletId": "store_id", "MedicineName": "product",
                   "UnitsInStock": "quantity", "SyncVersion": "update_id",
                   "confidence": 0.96},
        "note": "PascalCase — SyncVersion maps to update_id (version number)",
    },
    {
        "input":  ["branchName", "productDescription", "availableQty",
                   "lastSyncTime", "recordId"],
        "output": {"branchName": "store_id", "productDescription": "product",
                   "availableQty": "quantity", "lastSyncTime": "timestamp",
                   "recordId": "update_id", "confidence": 0.97},
        "note": "camelCase with full word descriptions",
    },

    # ---- ALLCAPS database / legacy system exports ---------------------------
    {
        "input":  ["BRANCH_CODE", "PRODUCT_DESC", "STOCK_LEVEL",
                   "REPORT_DT", "SEQ_NO"],
        "output": {"BRANCH_CODE": "store_id", "PRODUCT_DESC": "product",
                   "STOCK_LEVEL": "quantity", "REPORT_DT": "timestamp",
                   "SEQ_NO": "update_id", "confidence": 0.97},
        "note": "ALLCAPS SQL column style — SEQ_NO=sequence number=update_id",
    },
    {
        "input":  ["STORE_REF", "DRUG_CODE", "QTY_ON_HAND", "ENTRY_DATE"],
        "output": {"STORE_REF": "store_id", "DRUG_CODE": "product",
                   "QTY_ON_HAND": "quantity", "ENTRY_DATE": "timestamp",
                   "confidence": 0.96},
        "note": "ALLCAPS with REF suffix and ON_HAND stock pattern",
    },

    # ---- ERP / POS system column names -------------------------------------
    {
        "input":  ["SiteRef", "ItemDescription", "OnHandQty", "LastTxnDate"],
        "output": {"SiteRef": "store_id", "ItemDescription": "product",
                   "OnHandQty": "quantity", "LastTxnDate": "timestamp",
                   "confidence": 0.95},
        "note": "SAP/Oracle POS style — SiteRef=site reference=store",
    },
    {
        "input":  ["outlet_code", "sku_name", "closing_stock",
                   "report_generated_at", "batch_ref"],
        "output": {"outlet_code": "store_id", "sku_name": "product",
                   "closing_stock": "quantity",
                   "report_generated_at": "timestamp",
                   "batch_ref": "update_id", "confidence": 0.95},
        "note": "closing_stock=end-of-day inventory, batch_ref=update reference",
    },
    {
        "input":  ["facility", "sku", "inventory", "captured_on"],
        "output": {"facility": "store_id", "sku": "product",
                   "inventory": "quantity", "captured_on": "timestamp",
                   "confidence": 0.94},
        "note": "facility=store site, sku=product code, inventory=quantity",
    },

    # ---- Regional / transliterated names -----------------------------------
    {
        "input":  ["dawakhana", "dawa", "matra", "samay"],
        "output": {"dawakhana": "store_id", "dawa": "product",
                   "matra": "quantity", "samay": "timestamp",
                   "confidence": 0.88},
        "note": (
            "Hindi transliteration: dawakhana=pharmacy, dawa=medicine, "
            "matra=quantity/dose, samay=time"
        ),
    },
    {
        "input":  ["kedai", "ubat", "stok", "masa", "id_rekod"],
        "output": {"kedai": "store_id", "ubat": "product",
                   "stok": "quantity", "masa": "timestamp",
                   "id_rekod": "update_id", "confidence": 0.87},
        "note": (
            "Malay: kedai=shop, ubat=medicine, stok=stock, "
            "masa=time, id_rekod=record id"
        ),
    },

    # ---- "no. of X" and count-phrase quantity patterns ----------------------
    # These are the most commonly missed patterns because the rule-based
    # normaliser turns dots into underscores ("no. of pdts" -> "no__of_pdts")
    # and the result does not match any short synonym.
    {
        "input":  ["no. of pdts", "branch", "medicine name"],
        "output": {"no. of pdts": "quantity", "branch": "store_id",
                   "medicine name": "product", "confidence": 0.95},
        "note": (
            "no. of pdts = number of products = quantity. "
            "Dots and spaces in the column name must be kept verbatim as the key."
        ),
    },
    {
        "input":  ["no. of units", "store", "drug"],
        "output": {"no. of units": "quantity", "store": "store_id",
                   "drug": "product", "confidence": 0.96},
        "note": "no. of units = stock count = quantity",
    },
    {
        "input":  ["no of packs", "outlet", "item name", "date"],
        "output": {"no of packs": "quantity", "outlet": "store_id",
                   "item name": "product", "date": "timestamp",
                   "confidence": 0.95},
        "note": "no of packs (without dot) = quantity; packs = stock units",
    },
    {
        "input":  ["number of units available", "branch identifier",
                   "medicine name", "date and time of report"],
        "output": {"branch identifier": "store_id",
                   "medicine name": "product",
                   "number of units available": "quantity",
                   "date and time of report": "timestamp",
                   "confidence": 0.95},
        "note": "Full phrase quantity column — units available = stock level",
    },
    {
        "input":  ["total units on hand", "pharmacy branch", "drug name"],
        "output": {"pharmacy branch": "store_id", "drug name": "product",
                   "total units on hand": "quantity", "confidence": 0.96},
        "note": "total units on hand = inventory quantity",
    },
    {
        "input":  ["packs available", "facility code", "product name",
                   "last updated", "entry no"],
        "output": {"facility code": "store_id", "product name": "product",
                   "packs available": "quantity", "last updated": "timestamp",
                   "entry no": "update_id", "confidence": 0.94},
        "note": "packs available = quantity; entry no = update_id",
    },

    # ---- Ambiguous but inferrable columns ----------------------------------
    {
        "input":  ["ref", "description", "balance", "version"],
        "output": {"ref": "store_id", "description": "product",
                   "balance": "quantity", "version": "update_id",
                   "confidence": 0.82},
        "note": (
            "Generic names — ref=reference/store, balance=stock balance, "
            "version=record version number"
        ),
    },
    {
        "input":  ["outlet_name", "active_ingredient", "units",
                   "logged_at", "record_num"],
        "output": {"outlet_name": "store_id", "active_ingredient": "product",
                   "units": "quantity", "logged_at": "timestamp",
                   "record_num": "update_id", "confidence": 0.93},
        "note": "active_ingredient maps to product in pharmacy context",
    },
    {
        "input":  ["location_code", "formulation", "packs_available",
                   "sync_date", "entry_no"],
        "output": {"location_code": "store_id", "formulation": "product",
                   "packs_available": "quantity", "sync_date": "timestamp",
                   "entry_no": "update_id", "confidence": 0.93},
        "note": "formulation=drug formulation=product name, packs=quantity",
    },
]


def _format_few_shot_examples(unmapped_columns: List[str]) -> str:
    """
    Select the most relevant examples from _FEW_SHOT_EXAMPLES and format
    them as numbered INPUT / OUTPUT demonstration pairs.

    Selection strategy
    ------------------
    1. Score each example by how many of its input column names overlap
       (case-insensitively) with the current unmapped columns.
       Direct overlaps are the most instructive demonstrations.
    2. Fill remaining slots with structurally diverse examples so the LLM
       always sees a spread of naming styles (abbreviations, long names,
       camelCase, ALLCAPS, regional).
    3. Cap at MAX_EXAMPLES to keep the prompt inside a sensible token budget.
    """
    MAX_EXAMPLES = 6

    current_normalised = {c.strip().lower() for c in unmapped_columns}

    def overlap_score(ex: dict) -> int:
        ex_normalised = {c.strip().lower() for c in ex["input"]}
        return len(current_normalised & ex_normalised)

    ranked = sorted(
        enumerate(_FEW_SHOT_EXAMPLES),
        key=lambda t: (-overlap_score(t[1]), t[0]),
    )
    selected = [ex for _, ex in ranked[:MAX_EXAMPLES]]

    lines = []
    for i, ex in enumerate(selected, start=1):
        import json as _json
        output_str = _json.dumps(ex["output"], ensure_ascii=False)
        lines.append(
            f"  Example {i}  [{ex['note']}]\n"
            f"  Columns : {ex['input']}\n"
            f"  Output  : {output_str}"
        )
    return "\n\n".join(lines)


def _build_prompt(unmapped_columns: List[str]) -> str:
    """
    Build the full prompt sent to the LLM for schema mapping.

    Structure
    ---------
    1. Role and task statement
    2. Standard field definitions with plain-English meanings
    3. Numbered list of the exact columns to map in this call
    4. Curated few-shot examples drawn from _FEW_SHOT_EXAMPLES
    5. Strict output rules (repeated constraints improve compliance)
    6. A dynamic tail example built from the first real column name so
       the LLM cannot pattern-match to a static placeholder key
    """
    numbered = "\n".join(
        f'  {i + 1}. "{col}"' for i, col in enumerate(unmapped_columns)
    )
    few_shot_block = _format_few_shot_examples(unmapped_columns)

    first_col = unmapped_columns[0] if unmapped_columns else "column_name_here"
    tail_example = f'{{"{first_col}": "<standard_field>", "confidence": 0.95}}'

    return (
        "You are a data schema mapping assistant for a pharmacy inventory "
        "management system.\n\n"

        "TASK\n"
        "Map each column name in the COLUMNS TO MAP section to exactly one "
        "of the standard schema fields defined below.\n\n"

        "STANDARD SCHEMA FIELDS AND THEIR MEANINGS\n"
        f"  store_id  : The identifier for a pharmacy branch, shop, outlet, "
        "location, facility, or site. Could be a code, name, or reference number.\n"
        "  product   : The name, description, or code of a medicine, drug, item, "
        "SKU, active ingredient, or any pharmaceutical product.\n"
        "  quantity  : A numeric stock level — units on hand, available count, "
        "balance, inventory amount, number of packs, or closing stock.\n"
        "  timestamp : A date or datetime indicating when the stock level was "
        "recorded, captured, logged, reported, or last updated.\n"
        "  update_id : A unique identifier for this specific record — a "
        "transaction id, row id, record number, version, sequence, or batch "
        "reference.\n\n"

        f"Required fields (must be present in output): "
        f"{', '.join(sorted(REQUIRED_FIELDS))}\n\n"

        "COLUMNS TO MAP  (use these strings VERBATIM as your JSON keys — "
        "copy them character-for-character)\n"
        f"{numbered}\n\n"

        "FEW-SHOT EXAMPLES  (study these carefully before responding)\n"
        f"{few_shot_block}\n\n"

        "OUTPUT RULES  (follow every rule exactly)\n"
        "  1. Return ONLY a valid JSON object — no prose, no markdown, "
        "no code fences, no explanation.\n"
        "  2. Every KEY in your JSON must be copied VERBATIM from the "
        "COLUMNS TO MAP list above.\n"
        "     Do NOT shorten, abbreviate, translate, or paraphrase any key.\n"
        "     Do NOT include a key that does not appear in the list.\n"
        f"  3. Every VALUE must be one of: {', '.join(STANDARD_FIELDS)}\n"
        "  4. Include a 'confidence' key (float 0.0 to 1.0) for your overall "
        "certainty across all mappings.\n"
        "  5. Omit any column you cannot map with reasonable confidence — "
        "do not guess randomly.\n\n"

        "YOUR RESPONSE (replace <standard_field> with the correct field name):\n"
        f"{tail_example}"
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


def _validate_and_clean_mapping(
    mapping: Dict[str, str],
    original_columns: List[str],
) -> Tuple[Dict[str, str], List[str]]:
    """
    Validate LLM output and return (clean_mapping, rejection_reasons).

    Design change from the original all-or-nothing approach:
      The original implementation rejected the entire mapping if any single
      key was hallucinated. This meant one bad key could silently discard
      three correctly-mapped columns.

      This version filters out individual bad entries and returns whatever
      is valid. The caller decides whether the remaining mappings are
      sufficient. Rejection reasons are collected for logging/display.

    Acceptance rules per entry:
      1. The key must exist in original_columns (exact match or
         case/whitespace-insensitive match).
      2. The value must be one of STANDARD_FIELDS.
    """
    if not mapping:
        return {}, ["LLM returned an empty mapping."]

    # Build a case-insensitive + strip-normalised lookup so that minor
    # casing differences between the LLM key and the actual column name
    # do not cause a valid mapping to be discarded.
    orig_normalised: Dict[str, str] = {
        c.strip().lower(): c for c in original_columns
    }

    clean: Dict[str, str] = {}
    rejected: List[str] = []

    for col, field in mapping.items():
        # Check 1: value is a known standard field
        if field not in STANDARD_FIELDS:
            rejected.append(
                f"  Discarded '{col}' -> '{field}': '{field}' is not a standard field."
            )
            continue

        # Check 2: key exists in the file (exact or normalised match)
        exact_match = col in original_columns
        normalised_key = col.strip().lower()
        normalised_match = normalised_key in orig_normalised

        if exact_match:
            clean[col] = field
        elif normalised_match:
            # Use the canonical column name from the file, not the LLM's
            # version (which may differ in casing or surrounding whitespace)
            canonical = orig_normalised[normalised_key]
            clean[canonical] = field
        else:
            rejected.append(
                f"  Discarded '{col}' -> '{field}': "
                f"'{col}' does not match any column in the file."
            )

    return clean, rejected


def _validate_mapping(mapping: Dict[str, str], original_columns: List[str]) -> Tuple[bool, str]:
    """
    Legacy shim kept for any external callers.
    Delegates to _validate_and_clean_mapping and returns (True, "ok") if
    at least one valid entry survives, otherwise (False, first_rejection_reason).
    """
    clean, rejected = _validate_and_clean_mapping(mapping, original_columns)
    if clean:
        return True, "ok"
    reason = rejected[0] if rejected else "No valid mappings found."
    return False, reason


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


def clear_mapping_cache(cache_file: str = "data/llm_cache.json") -> None:
    """
    Flush both the in-memory and on-disk LLM mapping caches.

    Must clear both layers because _load_cache() reloads from disk
    whenever _memory_cache is empty.  Clearing only RAM means the bad
    result is reloaded from disk on the very next call, making the
    toggle appear to have no effect.

    Called when:
      - The user toggles LLM on or off in the sidebar
      - The app detects a new file upload (different identity)
    """
    global _memory_cache
    _memory_cache = {}

    abs_path = _resolve(cache_file)
    try:
        if os.path.isfile(abs_path):
            with open(abs_path, "w", encoding="utf-8") as fh:
                fh.write("{}")
    except Exception:
        pass


# Keep the old name as an alias so any external callers are not broken.
clear_memory_cache = clear_mapping_cache


def _resolve(path: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, path)