"""
app.py

Entry point for the Pharmacy Smart Sync Scheduler Streamlit application.

Run with:
    streamlit run app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

import streamlit as st

from core.conflict_resolver import ConflictResolver
from core.data_loader import load_file
from core.llm_mapper import llm_map_columns
from core.logger import get_logger
from core.retry_handler import RetryHandler
from core.scheduler import SyncScheduler
from core.schema_mapper import (
    apply_mapping,
    load_mapping_rules,
    rule_based_map,
    REQUIRED_FIELDS,
)
from core.sync_engine import SyncEngine
from core.utils import (
    database_to_df,
    dead_letter_to_df,
    load_settings,
    mapping_to_df,
    processed_to_df,
    queue_to_df,
    records_to_df,
)
from core.validator import validate_and_normalize

# ---------------------------------------------------------------------------
# Path helper  (defined here so it is available to _init_session)
# ---------------------------------------------------------------------------

def _resolve(path: str) -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, path)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Pharmacy Sync Scheduler",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] { min-width: 310px; }

        /* Metric cards - visible on both light and dark themes */
        div[data-testid="metric-container"] {
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 8px;
            padding: 12px 16px;
            background: rgba(255,255,255,0.06);
        }

        /* Section titles - neutral white so readable on dark background */
        .section-title {
            font-size: 1rem;
            font-weight: 600;
            color: #e2e8f0;
            border-bottom: 2px solid rgba(255,255,255,0.2);
            padding-bottom: 4px;
            margin-top: 1.2rem;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------

def _init_session() -> None:
    if "ready" in st.session_state:
        return

    settings = load_settings("config/settings.json")
    log_file = _resolve(settings.get("log_file", "data/sync.log"))
    logger = get_logger("sync_scheduler", log_file=log_file)

    scheduler = SyncScheduler(
        max_sync_per_window=settings["max_sync_per_window"],
        strategy=settings.get("scheduling_strategy", "fifo"),
        logger=logger,
    )
    resolver = ConflictResolver(
        policy=settings.get("conflict_policy", "multi_level"),
        logger=logger,
    )
    retry_handler = RetryHandler(
        retry_limit=settings.get("retry_limit", 3),
        logger=logger,
    )
    engine = SyncEngine(
        scheduler=scheduler,
        resolver=resolver,
        retry_handler=retry_handler,
        idempotency_store_path=_resolve(
            settings.get("idempotency_store_path", "data/processed_ids.json")
        ),
        logger=logger,
    )

    # Window checker — reads sync_schedule from settings
    from core.window_checker import WindowChecker
    schedule_cfg = settings.get("sync_schedule", {})
    window_checker = WindowChecker(schedule_cfg, logger=logger)

    st.session_state.settings = settings
    st.session_state.logger = logger
    st.session_state.scheduler = scheduler
    st.session_state.resolver = resolver
    st.session_state.retry_handler = retry_handler
    st.session_state.engine = engine
    st.session_state.window_checker = window_checker
    st.session_state.emergency_override = False

    # Upload pipeline state
    st.session_state.raw_df = None
    st.session_state.normalized_df = None
    st.session_state.file_info = {}
    st.session_state.mapping_result = {}
    st.session_state.mapping_sources = {}
    st.session_state.llm_confidence = None
    st.session_state.llm_reason = ""
    st.session_state.validation_warnings = []
    st.session_state.ready_to_enqueue = False

    # Session-level counter for total enqueued
    st.session_state.total_enqueued = 0
    st.session_state.upload_error = ""
    st.session_state.last_uploaded_identity = None

    # Notifications buffer
    st.session_state.notifications = []

    st.session_state.ready = True


_init_session()

# Aliases for brevity
engine: SyncEngine = st.session_state.engine
scheduler: SyncScheduler = st.session_state.scheduler
retry_handler: RetryHandler = st.session_state.retry_handler
settings: dict = st.session_state.settings
logger = st.session_state.logger

# Rebuild window_checker on every rerun so schedule changes in the UI
# take effect immediately without restarting the app.
from core.window_checker import WindowChecker as _WC, save_schedule_config, validate_window_entry
from core.llm_mapper import clear_mapping_cache as _clear_llm_cache
_schedule_cfg = settings.get("sync_schedule", {})
window_checker = _WC(_schedule_cfg, logger=logger)
st.session_state.window_checker = window_checker

# Trigger a genuine browser-initiated rerun every 10 seconds so the
# sync window countdown, queue metrics and sidebar numbers stay live.
# st_autorefresh injects a JS timer — it does NOT call st.rerun() at
# module level on first render, so it cannot interrupt file processing.
st_autorefresh(interval=10_000, limit=None, key="global_autorefresh")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title(settings["app_title"])
    st.caption(settings["app_subtitle"])
    st.divider()

    st.markdown("**Active Configuration**")
    cfg1, cfg2 = st.columns(2)
    cfg1.metric("Max / Window", settings["max_sync_per_window"])
    cfg2.metric("Strategy", settings.get("scheduling_strategy", "fifo").upper())
    cfg3, cfg4 = st.columns(2)
    cfg3.metric("Conflict Policy", settings.get("conflict_policy", "multi_level").replace("_", " ").title())
    cfg4.metric("Retry Limit", settings.get("retry_limit", 3))
    st.caption(f"Default TZ: {settings.get('default_timezone', 'UTC')}")

    llm_cfg = settings.get("llm", {})
    llm_currently_on = llm_cfg.get("enabled", False)
    llm_toggle = st.toggle(
        "LLM semantic mapping",
        value=llm_currently_on,
        help=(
            "When on, columns that rule-based matching cannot identify are sent "
            "to the configured LLM endpoint for semantic inference. "
            "Requires a valid endpoint and API key in config/settings.json."
        ),
    )
    if llm_toggle != llm_currently_on:
        llm_cfg["enabled"] = llm_toggle
        settings["llm"] = llm_cfg
        st.session_state.settings = settings
        try:
            import json as _json
            cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "settings.json")
            with open(cfg_path, "r", encoding="utf-8") as _fh:
                _disk = _json.load(_fh)
            _disk["llm"] = llm_cfg
            with open(cfg_path, "w", encoding="utf-8") as _fh:
                _json.dump(_disk, _fh, indent=4)
        except Exception:
            pass
        # Flush both RAM and disk LLM cache so a previously bad or
        # partial mapping cannot be reloaded from disk after the toggle.
        _clear_llm_cache(
            settings.get("llm", {}).get("cache_file", "data/llm_cache.json")
        )
        st.session_state.last_uploaded_identity = None
        st.rerun()
    st.caption("Edit config/settings.json for endpoint and API key settings.")
    st.divider()

    st.markdown("**Session Summary**")
    st.write(f"Total enqueued : **{st.session_state.total_enqueued}**")
    st.write(f"Pending in queue : **{scheduler.queue_size()}**")
    st.write(f"Sync cycles run : **{len(engine.sync_cycles)}**")
    st.write(f"DB entries : **{len(engine.get_database_records())}**")
    st.write(f"Dead letter queue : **{len(retry_handler.get_dead_letter_queue())}**")
    st.divider()

    with st.expander("Danger Zone"):
        if st.button("Reset session state", use_container_width=True):
            engine.reset()
            st.session_state.total_enqueued = 0
            st.session_state.raw_df = None
            st.session_state.normalized_df = None
            st.session_state.file_info = {}
            st.session_state.mapping_result = {}
            st.session_state.mapping_sources = {}
            st.session_state.llm_confidence = None
            st.session_state.llm_reason = ""
            st.session_state.validation_warnings = []
            st.session_state.ready_to_enqueue = False
            st.session_state.notifications = []
            st.session_state.upload_error = ""
            st.session_state.last_uploaded_identity = None
            st.rerun()


# ---------------------------------------------------------------------------
# Flush notifications
# ---------------------------------------------------------------------------

for note in st.session_state.notifications:
    st.success(note)
st.session_state.notifications.clear()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

(
    tab_upload,
    tab_dashboard,
    tab_queue,
    tab_history,
    tab_database,
    tab_dlq,
    tab_schedule,
) = st.tabs([
    "Upload and Normalize",
    "Dashboard",
    "Queue",
    "Sync History",
    "Central Database",
    "Dead Letter Queue",
    "Sync Schedule",
])


# ============================================================
# TAB 1 - Upload and Normalize
# ============================================================

with tab_upload:
    st.markdown("### Upload Stock Update File")
    st.caption(
        "Upload a CSV, XLSX, or JSON file. Required columns: store_id, product, quantity. "
        "update_id is optional — auto-generated if absent. "
        "timestamp is optional — if omitted the server records the upload time automatically. "
        "Column names are mapped using rule-based matching (and optionally LLM inference)."
    )

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls", "json"],
        help="Accepted formats: CSV, XLSX, XLS, JSON",
    )

    # -------------------------------------------------------------------
    # Processing gate.
    #
    # Identity is (filename, file size in bytes).  Using size alongside
    # name means re-uploading a different file with the same name still
    # triggers a full re-process, while auto-refresh reruns (where
    # uploaded_file is the same object) are correctly skipped.
    #
    # All results land in session state and survive every subsequent
    # rerun (including the 10-second autorefresh) without re-processing.
    # -------------------------------------------------------------------
    if uploaded_file is not None:
        file_identity = (uploaded_file.name, uploaded_file.size)
        last_identity = st.session_state.get("last_uploaded_identity")

        if file_identity != last_identity:
            # Clear all downstream state from any previous file
            st.session_state.raw_df = None
            st.session_state.normalized_df = None
            st.session_state.file_info = {}
            st.session_state.mapping_result = {}
            st.session_state.mapping_sources = {}
            st.session_state.llm_confidence = None
            st.session_state.llm_reason = ""
            st.session_state.validation_warnings = []
            st.session_state.ready_to_enqueue = False
            st.session_state.upload_error = ""
            st.session_state.last_uploaded_identity = file_identity

            # Step 1 — load raw file into DataFrame
            with st.spinner("Loading file..."):
                try:
                    raw_df, file_info = load_file(
                        uploaded_file, uploaded_file.name, logger=logger
                    )
                    st.session_state.raw_df = raw_df
                    st.session_state.file_info = file_info
                except Exception as exc:
                    st.session_state.upload_error = f"Could not load file: {exc}"
                    logger.error(f"File load failed: {exc}")

            if not st.session_state.upload_error and st.session_state.raw_df is not None:
                raw_df = st.session_state.raw_df
                rules = load_mapping_rules("config/schema_mapping.json")
                original_columns = list(raw_df.columns)

                # Step 2 — rule-based schema mapping
                rule_mapping, unmapped = rule_based_map(
                    original_columns, rules, logger=logger
                )
                combined_mapping: dict = dict(rule_mapping)
                mapping_sources: dict = {col: "rule-based" for col in rule_mapping}

                # Step 3 — LLM mapping for any columns still unmapped
                llm_conf = None
                llm_reason = ""
                if unmapped:
                    llm_config = settings.get("llm", {})
                    if llm_config.get("enabled", False):
                        with st.spinner("Calling LLM for semantic column mapping..."):
                            llm_map, llm_conf, llm_reason = llm_map_columns(
                                original_columns, llm_config, combined_mapping,
                                logger=logger
                            )
                        if llm_map:
                            for col, field in llm_map.items():
                                if col not in combined_mapping:
                                    combined_mapping[col] = field
                                    mapping_sources[col] = "llm"
                    else:
                        llm_reason = (
                            "LLM is disabled. Enable it in config/settings.json "
                            "for semantic fallback mapping."
                        )

                st.session_state.mapping_result = combined_mapping
                st.session_state.mapping_sources = mapping_sources
                st.session_state.llm_confidence = llm_conf
                st.session_state.llm_reason = llm_reason

                # Step 4 — validate only when all required fields are covered
                mapped_standards = set(combined_mapping.values())
                missing_required = {"store_id", "product", "quantity"} - mapped_standards

                if not missing_required:
                    mapped_df = apply_mapping(raw_df, combined_mapping)
                    with st.spinner("Validating and normalising data..."):
                        try:
                            normalized_df, val_warnings = validate_and_normalize(
                                mapped_df,
                                default_timezone=settings.get("default_timezone", "UTC"),
                                logger=logger,
                            )
                            st.session_state.normalized_df = normalized_df
                            st.session_state.validation_warnings = val_warnings
                            st.session_state.ready_to_enqueue = len(normalized_df) > 0
                        except ValueError as exc:
                            st.session_state.upload_error = str(exc)
                            st.session_state.ready_to_enqueue = False

    # -------------------------------------------------------------------
    # Display section: renders entirely from session state so it remains
    # visible across every auto-refresh rerun, not just the upload rerun.
    # -------------------------------------------------------------------

    if st.session_state.get("upload_error"):
        st.error(st.session_state.upload_error)

    elif st.session_state.raw_df is not None:
        fi = st.session_state.file_info
        raw_df = st.session_state.raw_df
        combined_mapping = st.session_state.mapping_result
        mapping_sources = st.session_state.mapping_sources
        llm_conf = st.session_state.llm_confidence
        llm_reason = st.session_state.llm_reason

        # File summary banner
        st.info(
            f"File: **{fi['name']}**  |  Format: {fi['format'].upper()}  |  "
            f"Rows loaded: **{fi['rows']}**  |  Columns detected: **{len(fi['columns'])}**  |  "
            f"Server stamps received_at on enqueue."
        )

        # ---- Schema mapping results ----
        st.markdown('<p class="section-title">Schema Mapping</p>', unsafe_allow_html=True)

        map_df = mapping_to_df(combined_mapping, mapping_sources)
        if not map_df.empty:
            st.dataframe(map_df, use_container_width=True, hide_index=True)
        else:
            st.warning(
                "No columns could be mapped. Check that the file contains "
                "recognisable column names."
            )

        still_unmapped = [c for c in fi["columns"] if c not in combined_mapping]
        if still_unmapped:
            st.warning(f"Unmapped columns (will be dropped): {', '.join(still_unmapped)}")

        llm_enabled = settings.get("llm", {}).get("enabled", False)
        still_unmapped_after_llm = [c for c in fi["columns"] if c not in combined_mapping]

        if llm_conf is not None:
            st.success(
                f"LLM semantic mapping applied  |  Confidence: {llm_conf:.2f}  |  {llm_reason}"
            )
        elif llm_enabled and not still_unmapped_after_llm:
            st.caption(f"LLM ran but all columns were already covered by rule-based mapping.")
        elif llm_enabled:
            st.caption(f"LLM ran  |  {llm_reason}")
        elif still_unmapped:
            # LLM is off and there are unmapped columns — show an actionable warning
            st.warning(
                f"{len(still_unmapped)} column(s) could not be mapped by rule-based matching "
                f"({', '.join(still_unmapped)}). "
                "Turn on **LLM semantic mapping** in the sidebar to attempt automatic inference, "
                "then the file will be re-processed immediately."
            )
        else:
            st.caption("Rule-based mapping resolved all columns. LLM was not needed.")

        # ---- Required-field coverage check ----
        mapped_standards = set(combined_mapping.values())
        truly_required = {"store_id", "product", "quantity"}
        missing_required = truly_required - mapped_standards

        if missing_required:
            found_cols = fi["columns"]
            st.error(
                f"Required fields still missing after mapping: "
                f"**{', '.join(sorted(missing_required))}**. "
                f"Columns found in this file: {', '.join(found_cols)}. "
                f"The system needs at minimum: store_id (or equivalent), "
                f"product (or equivalent), quantity (or equivalent). "
                + (
                    "Enable LLM semantic mapping in the sidebar — it may be able to "
                    "infer the correct mapping from your column names."
                    if not llm_enabled else
                    "LLM mapping ran but could not resolve the required fields. "
                    "This file may not be a pharmacy stock update file."
                )
            )

        else:
            # ---- Validation results ----
            st.markdown(
                '<p class="section-title">Validation Results</p>', unsafe_allow_html=True
            )

            for w in st.session_state.validation_warnings:
                st.warning(w)

            normalized_df = st.session_state.normalized_df
            n_valid = len(normalized_df) if normalized_df is not None else 0
            n_dropped = fi["rows"] - n_valid

            if n_valid > 0:
                count_c1, count_c2, count_c3, count_c4 = st.columns(4)
                count_c1.metric("Rows loaded", fi["rows"])
                count_c2.metric("Valid rows", n_valid)
                count_c3.metric("Rows dropped", n_dropped)
                count_c4.metric("Columns mapped", len(combined_mapping))

                st.success(
                    f"{n_valid} valid row(s) ready to enqueue. "
                    f"{n_dropped} row(s) dropped during validation."
                )

                # ---- Conflict pre-analysis ----
                # Show the user what conflict resolution will do before
                # records hit the sync engine, so they can inspect the
                # data before committing it to the queue.
                st.markdown(
                    '<p class="section-title">Conflict Pre-Analysis</p>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Two records conflict when they target the same (store, product) pair. "
                    "The sync engine resolves conflicts using three levels: "
                    "1) latest timestamp wins, "
                    "2) on equal timestamps the lexicographically higher update_id wins, "
                    "3) on equal update_ids the last record in arrival order wins. "
                    "Duplicate update_ids are dropped before conflict resolution runs."
                )

                if normalized_df is not None and not normalized_df.empty:
                    # Identify duplicate update_ids within this file
                    uid_counts = normalized_df["update_id"].value_counts()
                    duplicate_uids = uid_counts[uid_counts > 1].index.tolist()
                    n_duplicate_uids = len(duplicate_uids)

                    # Identify conflicting (store, product) pairs
                    pair_counts = normalized_df.groupby(
                        ["store_id", "product"]
                    ).size().reset_index(name="record_count")
                    conflicting_pairs = pair_counts[pair_counts["record_count"] > 1]
                    n_conflicts = len(conflicting_pairs)
                    n_conflict_records = int(
                        conflicting_pairs["record_count"].sum()
                    ) if n_conflicts > 0 else 0
                    n_conflict_will_drop = n_conflict_records - n_conflicts

                    ca1, ca2, ca3, ca4 = st.columns(4)
                    ca1.metric(
                        "Unique (store, product) pairs",
                        int(pair_counts["record_count"].gt(0).sum()),
                    )
                    ca2.metric(
                        "Conflicting pairs",
                        n_conflicts,
                        help="Same store + product appears more than once in this file.",
                    )
                    ca3.metric(
                        "Records that will be dropped by conflict resolution",
                        n_conflict_will_drop,
                    )
                    ca4.metric(
                        "Duplicate update_ids in this file",
                        n_duplicate_uids,
                        help="These are deduplicated before conflict resolution.",
                    )

                    if n_conflicts > 0:
                        with st.expander(
                            f"View {n_conflicts} conflicting (store, product) pair(s)"
                        ):
                            conflict_detail = conflicting_pairs.rename(columns={
                                "store_id": "Store",
                                "product": "Product",
                                "record_count": "Submissions in this file",
                            })
                            st.dataframe(
                                conflict_detail,
                                use_container_width=True,
                                hide_index=True,
                            )
                            st.caption(
                                "The sync engine will keep exactly one record per pair "
                                "using the resolution levels described above."
                            )

                    if n_duplicate_uids > 0:
                        with st.expander(f"View {n_duplicate_uids} duplicate update_id(s)"):
                            dup_detail = (
                                normalized_df[normalized_df["update_id"].isin(duplicate_uids)]
                                [["update_id", "store_id", "product", "quantity"]]
                                .sort_values("update_id")
                            )
                            st.dataframe(
                                dup_detail,
                                use_container_width=True,
                                hide_index=True,
                            )
                            st.caption(
                                "Only the last occurrence of each duplicate update_id "
                                "survives into conflict resolution."
                            )

            else:
                st.error("No valid rows remain after validation. Check the warnings above.")

            # ---- Data preview panels ----
            st.divider()
            col_raw, col_norm = st.columns(2)

            with col_raw:
                st.markdown(
                    '<p class="section-title">Raw Uploaded Data</p>',
                    unsafe_allow_html=True,
                )
                st.dataframe(raw_df.head(50), use_container_width=True, hide_index=True)
                if len(raw_df) > 50:
                    st.caption(f"Showing first 50 of {len(raw_df)} rows.")

            with col_norm:
                st.markdown(
                    '<p class="section-title">Normalised Data</p>',
                    unsafe_allow_html=True,
                )
                st.caption("received_at is stamped by the server at queue entry — not shown here.")
                if normalized_df is not None and not normalized_df.empty:
                    display_df = records_to_df(normalized_df.to_dict("records"))
                    st.dataframe(display_df.head(50), use_container_width=True, hide_index=True)
                    if len(display_df) > 50:
                        st.caption(f"Showing first 50 of {len(display_df)} rows.")
                else:
                    st.info("No valid rows to display.")

            # ---- Add to queue ----
            st.divider()
            if st.session_state.ready_to_enqueue:
                enqueue_col, info_col = st.columns([2, 5])
                with enqueue_col:
                    if st.button("Add to Queue", type="primary", use_container_width=True):
                        records = st.session_state.normalized_df.to_dict("records")
                        added = scheduler.enqueue(records)
                        st.session_state.total_enqueued += added
                        # Mark as enqueued so the button cannot be pressed twice
                        st.session_state.ready_to_enqueue = False
                        st.session_state.notifications.append(
                            f"Added {added} record(s) to the sync queue."
                        )
                        st.rerun()
                with info_col:
                    st.info(
                        f"{n_valid} record(s) will be added. "
                        f"Current queue size: {scheduler.queue_size()}."
                    )
            else:
                if st.session_state.file_info:
                    st.info(
                        "Records from this file have already been added to the queue, "
                        "or no valid rows remain. Upload a new file to enqueue more."
                    )

    else:
        st.info("Upload a file above to begin.")


# ============================================================
# TAB 2 - Dashboard
# ============================================================

with tab_dashboard:
    st.markdown("### Dashboard")

    # Metrics row 1
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Enqueued", st.session_state.total_enqueued)
    m2.metric("Pending", scheduler.queue_size())
    m3.metric("Processed", engine.metrics["total_processed"])
    m4.metric("Conflicts", engine.metrics["total_conflicts"])
    m5.metric("Failed", engine.metrics["total_failed"])
    m6.metric("Retries", engine.metrics["total_retries"])

    st.divider()

    # Window status banner
    wstatus = window_checker.get_status()
    enforce = settings.get("sync_schedule", {}).get("enforce_window", False)
    override = st.session_state.get("emergency_override", False)

    if not enforce:
        st.info(
            "Window enforcement is OFF. Processing is allowed at any time. "
            "Configure windows in the Sync Schedule tab."
        )
    elif wstatus.is_open:
        st.success(
            f"Window OPEN: **{wstatus.current_window.label}**   |   "
            f"Closes in **{wstatus.time_until_change_str()}**"
        )
    elif override:
        st.warning(
            "Emergency override is active. Processing bypasses the schedule. "
            "Disable it below once done."
        )
    else:
        next_label = wstatus.next_window.label if wstatus.next_window else "None"
        st.error(
            f"Window CLOSED   |   "
            f"Next: **{next_label}**   |   "
            f"Opens in **{wstatus.time_until_change_str()}**"
        )

    # Emergency override toggle (only shown when enforcement is on and window closed)
    if enforce and not wstatus.is_open:
        ov_col, _ = st.columns([2, 5])
        with ov_col:
            if override:
                if st.button("Disable Emergency Override", use_container_width=True):
                    st.session_state.emergency_override = False
                    st.rerun()
            else:
                if st.button(
                    "Emergency Override",
                    use_container_width=True,
                    help="Force processing outside the configured sync window. Use only for urgent situations.",
                ):
                    st.session_state.emergency_override = True
                    st.rerun()

    st.divider()

    # Determine if processing is allowed right now
    allowed, allow_reason = window_checker.processing_allowed(override=override)
    processing_blocked = scheduler.is_empty() or not allowed

    # Run sync window button
    btn_col, info_col = st.columns([2, 5])

    with btn_col:
        run_clicked = st.button(
            "Run Sync Window",
            type="primary",
            use_container_width=True,
            disabled=processing_blocked,
        )
    with info_col:
        if scheduler.is_empty():
            st.info("Queue is empty. Upload a file and add records to the queue first.")
        elif not allowed:
            st.warning(allow_reason)
        else:
            preview_n = min(scheduler.queue_size(), settings["max_sync_per_window"])
            current_win = wstatus.current_window.label if wstatus.current_window else "Override"
            st.write(
                f"Next window will process up to **{preview_n}** record(s) "
                f"({scheduler.queue_size()} remaining in queue).  "
                f"Window: **{current_win}**"
            )

    if run_clicked and allowed:
        current_label = (
            wstatus.current_window.label if wstatus.current_window else "Emergency Override"
        )
        with st.spinner("Running sync window..."):
            result = engine.run_sync_window(
                window_label=current_label,
                window_checker=window_checker,
            )
        if result["status"] == "empty":
            st.warning("Queue was empty when the window ran.")
        else:
            st.success(
                f"Cycle {result['cycle_number']} [{current_label}] complete: "
                f"applied {result['applied']}, "
                f"conflicts dropped {result['conflicts']}, "
                f"expired {result.get('expired', 0)}, "
                f"failed {result['failed']}, "
                f"retried {result['retried']}."
            )
        st.rerun()

    st.divider()

    # Per-cycle bar chart
    st.markdown('<p class="section-title">Sync Cycle Performance</p>', unsafe_allow_html=True)

    if not engine.sync_cycles:
        st.caption("No sync cycles run yet. The chart will appear here.")
    else:
        ok_cycles = [c for c in engine.sync_cycles if c["status"] == "ok"]
        if ok_cycles:
            cycle_nums = [c["cycle_number"] for c in ok_cycles]
            bar_fig = go.Figure()
            bar_fig.add_trace(go.Bar(
                name="Fetched", x=cycle_nums,
                y=[c["fetched"] for c in ok_cycles],
                marker_color="#4a90d9",
            ))
            bar_fig.add_trace(go.Bar(
                name="Applied", x=cycle_nums,
                y=[c["applied"] for c in ok_cycles],
                marker_color="#27ae60",
            ))
            bar_fig.add_trace(go.Bar(
                name="Conflicts Dropped", x=cycle_nums,
                y=[c["conflicts"] for c in ok_cycles],
                marker_color="#e74c3c",
            ))
            bar_fig.add_trace(go.Bar(
                name="Failed", x=cycle_nums,
                y=[c["failed"] for c in ok_cycles],
                marker_color="#f39c12",
            ))
            bar_fig.update_layout(
                barmode="group",
                xaxis_title="Sync Cycle",
                yaxis_title="Record Count",
                xaxis=dict(tickmode="linear", dtick=1, color="#c8d6e5"),
                yaxis=dict(color="#c8d6e5"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1, font=dict(color="#c8d6e5")),
                height=340,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=40, b=40, l=40, r=20),
                font=dict(color="#c8d6e5"),
            )
            bar_fig.update_xaxes(showgrid=False, linecolor="rgba(255,255,255,0.1)")
            bar_fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")
            st.plotly_chart(bar_fig, use_container_width=True)

    # Work completion donut and queue size over time
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<p class="section-title">Work Completion</p>', unsafe_allow_html=True)
        total_ever = st.session_state.total_enqueued
        if total_ever > 0:
            processed = engine.metrics["total_processed"]
            pending = scheduler.queue_size()
            donut = go.Figure(go.Pie(
                labels=["Processed", "Pending"],
                values=[processed, pending],
                hole=0.55,
                marker_colors=["#27ae60", "#f39c12"],
                textinfo="label+percent",
                hovertemplate="%{label}: %{value}<extra></extra>",
            ))
            donut.update_layout(
                height=260,
                margin=dict(t=20, b=20, l=20, r=20),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c8d6e5"),
            )
            st.plotly_chart(donut, use_container_width=True)
        else:
            st.caption("No records enqueued yet.")

    with chart_col2:
        st.markdown('<p class="section-title">Queue Size Over Cycles</p>', unsafe_allow_html=True)
        qs_history = engine.metrics["queue_size_history"]
        if qs_history:
            qs_cycles = [e["cycle"] for e in qs_history]
            qs_sizes = [e["size"] for e in qs_history]
            qs_fig = go.Figure(go.Scatter(
                x=qs_cycles,
                y=qs_sizes,
                mode="lines+markers",
                line=dict(color="#4a90d9", width=2),
                marker=dict(size=7),
                fill="tozeroy",
                fillcolor="rgba(74,144,217,0.15)",
            ))
            qs_fig.update_layout(
                xaxis_title="Cycle",
                yaxis_title="Pending Records",
                xaxis=dict(tickmode="linear", dtick=1, color="#c8d6e5"),
                yaxis=dict(color="#c8d6e5"),
                height=260,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=40, l=40, r=20),
                font=dict(color="#c8d6e5"),
            )
            qs_fig.update_xaxes(showgrid=False, linecolor="rgba(255,255,255,0.1)")
            qs_fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")
            st.plotly_chart(qs_fig, use_container_width=True)
        else:
            st.caption("Queue history will appear after the first sync cycle.")

    # Multi-cycle simulation
    st.divider()
    st.markdown('<p class="section-title">Multi-Cycle Simulation</p>', unsafe_allow_html=True)
    st.caption(
        "Run multiple sync windows automatically. If records fail during a cycle "
        "they are re-queued for retry, which is why the queue may not be fully "
        "empty after N cycles. Use Run Until Empty to fully drain everything."
    )

    sim_n_col, sim_btn_col, sim_empty_col = st.columns([2, 2, 2])

    with sim_n_col:
        sim_count = st.number_input(
            "Cycles to simulate", min_value=1, max_value=100, value=3, step=1
        )

    with sim_btn_col:
        st.write("")
        st.write("")
        if st.button(
            f"Simulate {sim_count} Cycle(s)",
            type="secondary",
            use_container_width=True,
            disabled=processing_blocked,
        ):
            if not allowed:
                st.warning(allow_reason)
                st.stop()
            retries_before = engine.metrics["total_retries"]
            completed = 0
            with st.spinner(f"Running {sim_count} sync cycle(s)..."):
                for _ in range(int(sim_count)):
                    if scheduler.is_empty():
                        break
                    _sim_label = (
                        wstatus.current_window.label
                        if wstatus.current_window else "Emergency Override"
                    )
                    engine.run_sync_window(
                        window_label=_sim_label,
                        window_checker=window_checker,
                    )
                    completed += 1

            retries_added = engine.metrics["total_retries"] - retries_before
            pending_now = scheduler.queue_size()

            if pending_now == 0:
                st.success(
                    f"{completed} cycle(s) run. Queue fully drained."
                )
            else:
                # Explain clearly why records are still pending
                st.warning(
                    f"{completed} cycle(s) run. "
                    f"**{pending_now} record(s) still pending** — "
                    f"these are retry attempts re-queued after failing "
                    f"({retries_added} retrie(s) issued in this run). "
                    f"They are not new original records. "
                    f"Run more cycles or click **Run Until Empty** to fully resolve them."
                )
            st.rerun()

    with sim_empty_col:
        st.write("")
        st.write("")
        if st.button(
            "Run Until Empty",
            type="primary",
            use_container_width=True,
            disabled=processing_blocked,
            help="Keeps running sync windows until the queue is completely empty, "
                 "including all retry attempts.",
        ):
            completed_all = 0
            MAX_SAFETY = 500  # prevent infinite loop if retry_limit is very high
            with st.spinner("Running until queue is empty..."):
                while not scheduler.is_empty() and completed_all < MAX_SAFETY:
                    _ue_label = (
                        wstatus.current_window.label
                        if wstatus.current_window else "Emergency Override"
                    )
                    engine.run_sync_window(
                        window_label=_ue_label,
                        window_checker=window_checker,
                    )
                    completed_all += 1

            dlq_count = len(engine.retry_handler.get_dead_letter_queue())
            if scheduler.is_empty():
                st.success(
                    f"Queue fully drained in {completed_all} cycle(s). "
                    f"{dlq_count} record(s) moved to Dead Letter Queue after "
                    f"exhausting retries."
                )
            else:
                st.warning(
                    f"Stopped after {MAX_SAFETY} cycles (safety limit). "
                    f"{scheduler.queue_size()} record(s) still pending."
                )
            st.rerun()


# ============================================================
# TAB 3 - Queue
# ============================================================

with tab_queue:
    st.markdown("### Queue Status")

    if scheduler.is_empty():
        st.info("The queue is empty.")
    else:
        pending = scheduler.get_all_pending()
        st.write(
            f"**{len(pending)}** record(s) are waiting. "
            f"Next window will pull up to **{settings['max_sync_per_window']}** (strategy: "
            f"**{settings.get('scheduling_strategy','fifo').upper()}**). "
            f"The **received_at** column shows when the server accepted each record."
        )

        next_n = min(settings["max_sync_per_window"], len(pending))

        st.markdown('<p class="section-title">Next Batch (highlighted)</p>', unsafe_allow_html=True)
        next_df = queue_to_df(pending[:next_n])

        def _next_batch_row(row):
            return ["background-color: #0e4f6e; color: #ffffff; font-weight: 600;"] * len(row)

        st.dataframe(
            next_df.style.apply(_next_batch_row, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        if len(pending) > next_n:
            remainder_df = queue_to_df(pending[next_n:])
            with st.expander(
                f"Remaining {len(pending) - next_n} record(s) (future windows)"
            ):
                st.dataframe(remainder_df, use_container_width=True, hide_index=True)

        # Fair scheduling stats
        if settings.get("scheduling_strategy") == "priority":
            store_counts = scheduler.store_counts()
            if store_counts:
                st.markdown('<p class="section-title">Fair Scheduling Stats (records dequeued per store)</p>', unsafe_allow_html=True)
                fc_df = pd.DataFrame(
                    [{"store_id": k, "records_dequeued": v} for k, v in sorted(store_counts.items())]
                )
                st.dataframe(fc_df, use_container_width=True, hide_index=True)


# ============================================================
# TAB 4 - Sync History
# ============================================================

with tab_history:
    st.markdown("### Sync Cycle History")

    if not engine.sync_cycles:
        st.info("No sync cycles have been run yet.")
    else:
        ok_cycles = [c for c in engine.sync_cycles if c["status"] == "ok"]
        if ok_cycles:
            ok_df_hist = pd.DataFrame(ok_cycles)
            show_hist = [c for c in
                ["cycle_number", "window_label", "timestamp", "fetched", "applied",
                 "conflicts", "duplicates", "expired", "failed", "retried"]
                if c in ok_df_hist.columns]
            cycles_display = ok_df_hist[show_hist].rename(columns={
                "cycle_number": "Cycle",
                "window_label": "Window",
                "timestamp": "Ran At (UTC)",
                "fetched": "Fetched",
                "applied": "Applied",
                "conflicts": "Conflicts",
                "duplicates": "Duplicates",
                "expired": "Expired",
                "failed": "Failed",
                "retried": "Retried",
            })
            st.dataframe(cycles_display, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown('<p class="section-title">All Processed Records (Audit Log)</p>', unsafe_allow_html=True)

        proc_df = processed_to_df(engine.processed_records)
        if not proc_df.empty:
            STATUS_COLORS = {
                "applied":           {"bg": "#1a7a4a", "fg": "#ffffff"},
                "conflict_dropped":  {"bg": "#b85c00", "fg": "#ffffff"},
                "duplicate_skipped": {"bg": "#1565c0", "fg": "#ffffff"},
                "failed":            {"bg": "#b71c1c", "fg": "#ffffff"},
            }

            def _colour_by_status(row):
                colours = STATUS_COLORS.get(str(row.get("sync_status", "")))
                if colours:
                    style = f"background-color: {colours['bg']}; color: {colours['fg']}; font-weight: 500;"
                    return [style] * len(row)
                return [""] * len(row)

            st.dataframe(
                proc_df.style.apply(_colour_by_status, axis=1),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Dark green = applied  |  Dark orange = conflict dropped  |  "
                "Dark blue = duplicate skipped  |  Dark red = failed"
            )
        else:
            st.info("No records have been processed yet.")


# ============================================================
# TAB 5 - Central Database
# ============================================================

with tab_database:
    st.markdown("### Central Stock Database")
    st.caption(
        "In-memory state of the central pharmacy stock database. "
        "Only the most recently synced, conflict-resolved value per "
        "(store, product) pair is stored."
    )

    db_records = engine.get_database_records()
    db_df = database_to_df(db_records)

    if db_df.empty:
        st.info("Database is empty. Run at least one sync window to populate it.")
    else:
        # Filter controls
        fc1, fc2 = st.columns(2)
        all_stores = sorted(db_df["store_id"].unique().tolist())
        all_products = sorted(db_df["product"].unique().tolist())

        with fc1:
            sel_stores = st.multiselect("Filter by store", all_stores, placeholder="All stores")
        with fc2:
            sel_products = st.multiselect("Filter by product", all_products, placeholder="All products")

        filtered = db_df.copy()
        if sel_stores:
            filtered = filtered[filtered["store_id"].isin(sel_stores)]
        if sel_products:
            filtered = filtered[filtered["product"].isin(sel_products)]

        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(filtered)} of {len(db_df)} entries.")

        # Heatmap
        st.divider()
        st.markdown('<p class="section-title">Stock Heatmap (Quantity by Store and Product)</p>', unsafe_allow_html=True)

        pivot = db_df.pivot_table(
            index="store_id", columns="product", values="quantity", aggfunc="first"
        )
        if not pivot.empty:
            heat_fig = go.Figure(go.Heatmap(
                z=pivot.values,
                x=list(pivot.columns),
                y=list(pivot.index),
                colorscale="Blues",
                text=pivot.values,
                texttemplate="%{text}",
                showscale=True,
                hovertemplate="Store: %{y}<br>Product: %{x}<br>Qty: %{z}<extra></extra>",
            ))
            heat_fig.update_layout(
                xaxis_title="Product",
                yaxis_title="Store",
                height=max(280, 100 + len(pivot.index) * 55),
                margin=dict(t=30, b=80, l=100, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c8d6e5"),
                xaxis=dict(color="#c8d6e5"),
                yaxis=dict(color="#c8d6e5"),
            )
            st.plotly_chart(heat_fig, use_container_width=True)


# ============================================================
# TAB 6 - Dead Letter Queue
# ============================================================

with tab_dlq:
    st.markdown("### Dead Letter Queue")
    st.caption(
        "Records land here after exhausting all configured retries. "
        "The failure reason explains why each record could not be applied. "
        "These records are never written to the central database."
    )

    dlq = retry_handler.get_dead_letter_queue()
    dlq_df = dead_letter_to_df(dlq)

    # Summary metrics
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("In Dead Letter Queue", len(dlq))
    rc2.metric("Total Retries Issued", engine.metrics["total_retries"])
    rc3.metric("Duplicates Skipped", engine.metrics["total_duplicates_skipped"])

    st.divider()

    if dlq_df.empty:
        st.success("Dead letter queue is empty. All records processed successfully.")
    else:
        st.warning(
            f"{len(dlq_df)} record(s) failed permanently and were not applied to the database."
        )

        # Colour rows by failure reason category
        def _colour_dlq_row(row):
            reason = str(row.get("failure_reason", "")).lower()
            if "stale" in reason or "out-of-order" in reason:
                style = "background-color: #b85c00; color: #ffffff; font-weight: 500;"
            elif "processing error" in reason or "rollback" in reason:
                style = "background-color: #b71c1c; color: #ffffff; font-weight: 500;"
            else:
                style = "background-color: #4a148c; color: #ffffff; font-weight: 500;"
            return [style] * len(row)

        st.dataframe(
            dlq_df.style.apply(_colour_dlq_row, axis=1),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Dark orange = stale / out-of-order update   |   "
            "Dark red = processing or rollback error   |   "
            "Dark purple = other"
        )

        # Failure reason breakdown
        st.divider()
        st.markdown('<p class="section-title">Failure Reason Breakdown</p>', unsafe_allow_html=True)

        if "failure_reason" in dlq_df.columns:
            def _categorise(reason: str) -> str:
                r = reason.lower()
                if "stale" in r or "out-of-order" in r:
                    return "Stale / out-of-order update"
                if "processing error" in r or "rollback" in r:
                    return "Processing / rollback error"
                return "Other"

            breakdown = (
                dlq_df["failure_reason"]
                .apply(_categorise)
                .value_counts()
                .reset_index()
                .rename(columns={"failure_reason": "Count", "index": "Category"})
            )
            # pandas value_counts().reset_index() column names vary by version
            if "failure_reason" in breakdown.columns:
                breakdown = breakdown.rename(columns={"failure_reason": "Category", "count": "Count"})
            st.dataframe(breakdown, use_container_width=True, hide_index=True)





# ============================================================
# TAB 7 - Sync Schedule Manager
# ============================================================

with tab_schedule:
    st.markdown("### Sync Schedule Manager")
    st.caption(
        "Configure the time windows during which the central server is allowed "
        "to process stock updates. All branches follow this schedule automatically. "
        "Changes take effect immediately without restarting the application."
    )

    schedule_cfg = settings.get("sync_schedule", {})
    enforce = schedule_cfg.get("enforce_window", False)
    tz_name = schedule_cfg.get("timezone", "UTC")
    expiry_h = schedule_cfg.get("update_expiry_hours", 48)
    raw_windows = schedule_cfg.get("windows", [])

    # ---- Live status banner ----
    live_status = window_checker.get_status()
    st.divider()
    st.markdown('<p class="section-title">Current Status</p>', unsafe_allow_html=True)

    status_c1, status_c2, status_c3 = st.columns(3)
    status_c1.metric(
        "Enforcement",
        "ON" if enforce else "OFF",
    )
    status_c2.metric("Active Window", live_status.current_window.label if live_status.is_open else "None")
    status_c3.metric(
        "Opens In" if not live_status.is_open else "Closes In",
        live_status.time_until_change_str(),
    )

    if enforce:
        if live_status.is_open:
            st.success(
                f"Window OPEN: **{live_status.current_window.label}**  |  "
                f"Closes in {live_status.time_until_change_str()}  |  "
                f"Timezone: {tz_name}"
            )
        else:
            next_l = live_status.next_window.label if live_status.next_window else "None configured"
            st.error(
                f"Window CLOSED  |  "
                f"Next: **{next_l}**  |  "
                f"Opens in {live_status.time_until_change_str()}  |  "
                f"Timezone: {tz_name}"
            )
    else:
        st.info("Enforcement is off. The application processes records at any time.")

    # ---- Global settings ----
    st.divider()
    st.markdown('<p class="section-title">Global Settings</p>', unsafe_allow_html=True)

    gs1, gs2, gs3 = st.columns(3)

    with gs1:
        new_enforce = st.toggle(
            "Enforce sync windows",
            value=enforce,
            help="When ON, the Run Sync Window and Simulate buttons are blocked "
                 "outside configured windows.",
        )
    with gs2:
        import pytz as _pytz
        common_tz = [
            "UTC", "Asia/Kolkata", "Asia/Kolkata", "America/New_York",
            "America/Los_Angeles", "Europe/London", "Europe/Paris",
            "Asia/Singapore", "Asia/Tokyo", "Australia/Sydney",
        ]
        # Deduplicate while preserving order
        seen = set()
        tz_options = [t for t in common_tz if not (t in seen or seen.add(t))]
        current_idx = tz_options.index(tz_name) if tz_name in tz_options else 0
        new_tz = st.selectbox(
            "Schedule timezone",
            options=tz_options,
            index=current_idx,
            help="All window start/end times are interpreted in this timezone.",
        )
    with gs3:
        new_expiry = st.number_input(
            "Update expiry (hours)",
            min_value=1,
            max_value=720,
            value=int(expiry_h),
            step=1,
            help="Updates older than this value are automatically moved to the "
                 "Dead Letter Queue when a sync window runs.",
        )

    # ---- Configured windows table ----
    st.divider()
    st.markdown('<p class="section-title">Configured Windows</p>', unsafe_allow_html=True)
    st.caption(
        "Edit the start and end times for any window directly in the row below. "
        "Press Save on that row to apply. Remove deletes the window entirely. "
        "Use Add New Window at the bottom to create additional slots."
    )

    if not raw_windows:
        st.info("No windows configured yet. Add one below.")
    else:
        import datetime as _dt_inline

        # Column header row
        h_label, h_start, h_end, h_save, h_remove = st.columns([3, 2, 2, 1, 1])
        h_label.markdown("**Window Label**")
        h_start.markdown("**Start (HH:MM)**")
        h_end.markdown("**End (HH:MM)**")

        st.divider()

        for row_idx, win in enumerate(raw_windows):
            col_label, col_start, col_end, col_save, col_remove = st.columns([3, 2, 2, 1, 1])

            with col_label:
                # Label is displayed read-only; rename via remove + re-add
                st.text_input(
                    "Label",
                    value=win.get("label", ""),
                    key=f"win_label_{row_idx}",
                    disabled=True,
                    label_visibility="collapsed",
                )

            with col_start:
                edited_start = st.text_input(
                    "Start",
                    value=win.get("start", ""),
                    key=f"win_start_{row_idx}",
                    placeholder="HH:MM",
                    label_visibility="collapsed",
                )

            with col_end:
                edited_end = st.text_input(
                    "End",
                    value=win.get("end", ""),
                    key=f"win_end_{row_idx}",
                    placeholder="HH:MM",
                    label_visibility="collapsed",
                )

            with col_save:
                st.write("")
                if st.button("Save", key=f"win_save_{row_idx}", use_container_width=True):
                    valid, err = validate_window_entry(win["label"], edited_start, edited_end)
                    if not valid:
                        st.error(err)
                    else:
                        overlap_found = False
                        try:
                            ns = _dt_inline.datetime.strptime(edited_start.strip(), "%H:%M").time()
                            ne = _dt_inline.datetime.strptime(edited_end.strip(), "%H:%M").time()
                            for other_idx, other_win in enumerate(raw_windows):
                                if other_idx == row_idx:
                                    continue
                                ws = _dt_inline.datetime.strptime(other_win["start"], "%H:%M").time()
                                we = _dt_inline.datetime.strptime(other_win["end"], "%H:%M").time()
                                if ns < we and ne > ws:
                                    st.error(
                                        f"Overlaps with '{other_win['label']}' "
                                        f"({other_win['start']} - {other_win['end']}). "
                                        "Windows must not overlap."
                                    )
                                    overlap_found = True
                                    break
                        except Exception:
                            pass

                        if not overlap_found:
                            updated_windows = list(raw_windows)
                            updated_windows[row_idx] = {
                                "label": win["label"],
                                "start": edited_start.strip(),
                                "end": edited_end.strip(),
                            }
                            updated_windows.sort(key=lambda x: x["start"])
                            new_cfg = {
                                "enforce_window": new_enforce,
                                "timezone": new_tz,
                                "update_expiry_hours": new_expiry,
                                "windows": updated_windows,
                            }
                            if save_schedule_config(new_cfg):
                                settings["sync_schedule"] = new_cfg
                                st.session_state.settings = settings
                                st.success(
                                    f"Saved: {win['label']} updated to "
                                    f"{edited_start.strip()} - {edited_end.strip()}."
                                )
                                st.rerun()
                            else:
                                st.error("Could not save changes. Check file permissions.")

            with col_remove:
                st.write("")
                if st.button("Remove", key=f"win_remove_{row_idx}", use_container_width=True):
                    updated_windows = [
                        w for idx, w in enumerate(raw_windows) if idx != row_idx
                    ]
                    new_cfg = {
                        "enforce_window": new_enforce,
                        "timezone": new_tz,
                        "update_expiry_hours": new_expiry,
                        "windows": updated_windows,
                    }
                    if save_schedule_config(new_cfg):
                        settings["sync_schedule"] = new_cfg
                        st.session_state.settings = settings
                        st.success(f"Removed window: {win['label']}")
                        st.rerun()
                    else:
                        st.error("Could not save changes. Check file permissions.")

    # ---- Add new window ----
    st.divider()
    st.markdown('<p class="section-title">Add New Window</p>', unsafe_allow_html=True)

    add_c1, add_c2, add_c3, add_c4 = st.columns([3, 2, 2, 2])
    with add_c1:
        new_label = st.text_input(
            "Window name",
            placeholder="e.g. Morning Sync",
            help="A descriptive name for this window.",
        )
    with add_c2:
        new_start = st.text_input(
            "Start time (HH:MM)",
            placeholder="06:00",
        )
    with add_c3:
        new_end = st.text_input(
            "End time (HH:MM)",
            placeholder="08:00",
        )
    with add_c4:
        st.write("")
        st.write("")
        add_clicked = st.button("Add Window", type="primary", use_container_width=True)

    if add_clicked:
        valid, err = validate_window_entry(new_label, new_start, new_end)
        if not valid:
            st.error(err)
        else:
            # Check for overlap with existing windows
            overlap = False
            import datetime as _dt
            try:
                ns = _dt.datetime.strptime(new_start.strip(), "%H:%M").time()
                ne = _dt.datetime.strptime(new_end.strip(), "%H:%M").time()
                for w in raw_windows:
                    ws = _dt.datetime.strptime(w["start"], "%H:%M").time()
                    we = _dt.datetime.strptime(w["end"], "%H:%M").time()
                    if ns < we and ne > ws:
                        overlap = True
                        st.error(
                            f"This window overlaps with '{w['label']}' "
                            f"({w['start']} - {w['end']}). "
                            "Windows must not overlap."
                        )
                        break
            except Exception:
                pass

            if not overlap:
                updated_windows = raw_windows + [
                    {"label": new_label.strip(), "start": new_start.strip(), "end": new_end.strip()}
                ]
                # Sort by start time
                updated_windows.sort(key=lambda w: w["start"])
                new_cfg = {
                    "enforce_window": new_enforce,
                    "timezone": new_tz,
                    "update_expiry_hours": new_expiry,
                    "windows": updated_windows,
                }
                if save_schedule_config(new_cfg):
                    settings["sync_schedule"] = new_cfg
                    st.session_state.settings = settings
                    st.success(
                        f"Added window: {new_label} ({new_start} - {new_end}). "
                        "Changes are live immediately."
                    )
                    st.rerun()
                else:
                    st.error("Could not save to settings.json. Check file permissions.")

    # ---- Save global settings (enforce / tz / expiry) ----
    st.divider()
    if st.button("Save Global Settings", type="secondary", use_container_width=False):
        new_cfg = {
            "enforce_window": new_enforce,
            "timezone": new_tz,
            "update_expiry_hours": new_expiry,
            "windows": raw_windows,
        }
        if save_schedule_config(new_cfg):
            settings["sync_schedule"] = new_cfg
            st.session_state.settings = settings
            st.success("Global settings saved. Changes are live immediately.")
            st.rerun()
        else:
            st.error("Could not save to settings.json. Check file permissions.")

    # ---- 24-hour timeline visualisation ----
    st.divider()
    st.markdown('<p class="section-title">24-Hour Timeline</p>', unsafe_allow_html=True)
    st.caption(
        "Grey = closed. Coloured bars = sync windows. "
        "Red line = current time in the configured timezone."
    )

    if not raw_windows:
        st.info("Add windows above to see the timeline.")
    else:
        import plotly.graph_objects as _go2

        timeline = _go2.Figure()

        # Full-day background bar
        timeline.add_trace(_go2.Bar(
            x=[24 * 60],
            y=["Schedule"],
            orientation="h",
            marker_color="rgba(255,255,255,0.08)",
            width=0.4,
            base=0,
            showlegend=False,
            hoverinfo="skip",
        ))

        colours = ["#1565c0", "#0e7c61", "#8e24aa", "#c62828", "#e65100", "#00695c"]
        for idx, w in enumerate(raw_windows):
            try:
                import datetime as _dt2
                ws = _dt2.datetime.strptime(w["start"], "%H:%M")
                we = _dt2.datetime.strptime(w["end"], "%H:%M")
                s_min = ws.hour * 60 + ws.minute
                dur = (we.hour * 60 + we.minute) - s_min
                col = colours[idx % len(colours)]
                timeline.add_trace(_go2.Bar(
                    x=[dur],
                    y=["Schedule"],
                    orientation="h",
                    marker_color=col,
                    width=0.4,
                    base=s_min,
                    name=w.get("label", f"Window {idx+1}"),
                    hovertemplate=f"{w.get('label', '')}:<br>{w['start']} - {w['end']}<extra></extra>",
                ))
            except Exception:
                continue

        # Current time marker
        import pytz as _pytz2
        import datetime as _dt3
        tz_obj = _pytz2.timezone(tz_name) if tz_name in _pytz2.all_timezones else _pytz2.UTC
        now_local = _dt3.datetime.now(tz=tz_obj)
        now_min = now_local.hour * 60 + now_local.minute

        timeline.add_vline(
            x=now_min,
            line_width=2,
            line_color="#ef5350",
            annotation_text=f"Now ({now_local.strftime('%H:%M')} {tz_name})",
            annotation_position="top right",
            annotation_font_color="#ef5350",
        )

        # X-axis: hours
        tick_vals = list(range(0, 24 * 60 + 1, 60))
        tick_text = [f"{h:02d}:00" for h in range(25)]

        timeline.update_layout(
            barmode="overlay",
            height=140,
            xaxis=dict(
                tickvals=tick_vals,
                ticktext=tick_text,
                range=[0, 24 * 60],
                color="#c8d6e5",
                showgrid=False,
            ),
            yaxis=dict(showticklabels=False, showgrid=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=30, b=30, l=10, r=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.1,
                xanchor="left",
                x=0,
                font=dict(color="#c8d6e5"),
            ),
            font=dict(color="#c8d6e5"),
        )
        st.plotly_chart(timeline, use_container_width=True)