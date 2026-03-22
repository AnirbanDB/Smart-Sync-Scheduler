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

    st.session_state.settings = settings
    st.session_state.logger = logger
    st.session_state.scheduler = scheduler
    st.session_state.resolver = resolver
    st.session_state.retry_handler = retry_handler
    st.session_state.engine = engine

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
    llm_status = "On" if settings.get("llm", {}).get("enabled", False) else "Off"
    st.caption(f"Default TZ: {settings.get('default_timezone', 'UTC')}  |  LLM: {llm_status}")
    st.caption("Edit `config/settings.json` then restart to change configuration.")
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
) = st.tabs([
    "Upload and Normalize",
    "Dashboard",
    "Queue",
    "Sync History",
    "Central Database",
    "Dead Letter Queue",
])


# ============================================================
# TAB 1 - Upload and Normalize
# ============================================================

with tab_upload:
    st.markdown("### Upload Stock Update File")
    st.caption(
        "Upload a CSV, XLSX, or JSON file. Column names are automatically mapped to "
        "the standard schema using rule-based matching (and optionally LLM inference). "
        "Valid records are added to the sync queue."
    )

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls", "json"],
        help="Accepted formats: CSV, XLSX, XLS, JSON",
    )

    if uploaded_file is not None:
        # Load raw file
        with st.spinner("Loading file..."):
            try:
                raw_df, file_info = load_file(
                    uploaded_file, uploaded_file.name, logger=logger
                )
                st.session_state.raw_df = raw_df
                st.session_state.file_info = file_info
                st.session_state.ready_to_enqueue = False
            except Exception as exc:
                st.error(f"Could not load file: {exc}")
                st.stop()

        fi = st.session_state.file_info
        st.info(
            f"File: **{fi['name']}**  |  Format: {fi['format'].upper()}  |  "
            f"Rows: {fi['rows']}  |  Columns detected: {len(fi['columns'])}"
        )

        # --- Schema Mapping ---
        st.markdown('<p class="section-title">Schema Mapping</p>', unsafe_allow_html=True)

        rules = load_mapping_rules("config/schema_mapping.json")
        original_columns = list(raw_df.columns)

        # Rule-based pass
        rule_mapping, unmapped = rule_based_map(original_columns, rules, logger=logger)
        combined_mapping: dict = dict(rule_mapping)
        mapping_sources: dict = {col: "rule-based" for col in rule_mapping}

        # LLM pass (if enabled and columns remain unmapped)
        llm_used = False
        llm_conf = None
        llm_reason = ""

        if unmapped:
            llm_config = settings.get("llm", {})
            if llm_config.get("enabled", False):
                with st.spinner("Calling LLM for semantic column mapping..."):
                    llm_map, llm_conf, llm_reason = llm_map_columns(
                        original_columns, llm_config, combined_mapping, logger=logger
                    )
                if llm_map:
                    for col, field in llm_map.items():
                        if col not in combined_mapping:
                            combined_mapping[col] = field
                            mapping_sources[col] = "llm"
                    llm_used = True
            else:
                llm_reason = "LLM is disabled. Enable it in config/settings.json."

        st.session_state.mapping_result = combined_mapping
        st.session_state.mapping_sources = mapping_sources
        st.session_state.llm_confidence = llm_conf
        st.session_state.llm_reason = llm_reason

        # Display mapping table
        map_df = mapping_to_df(combined_mapping, mapping_sources)
        if not map_df.empty:
            st.dataframe(map_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No columns could be mapped. Check that the file has recognisable column names.")

        # Unmapped columns warning
        still_unmapped = [c for c in original_columns if c not in combined_mapping]
        if still_unmapped:
            st.warning(
                f"Unmapped columns (will be dropped): {', '.join(still_unmapped)}"
            )

        # LLM status line
        if llm_used:
            st.caption(
                f"LLM mapping applied. Confidence: {llm_conf:.2f}  |  {llm_reason}"
            )
        elif settings.get("llm", {}).get("enabled", False):
            st.caption(f"LLM result: {llm_reason}")
        else:
            st.caption("LLM is off. Only rule-based mapping was applied.")

        # Check required fields are covered
        mapped_standards = set(combined_mapping.values())
        missing_required = REQUIRED_FIELDS - mapped_standards
        if missing_required:
            st.error(
                f"Required fields could not be mapped: {', '.join(sorted(missing_required))}. "
                "Cannot proceed. Rename columns or enable LLM mapping."
            )
        else:
            # --- Apply mapping and validate ---
            mapped_df = apply_mapping(raw_df, combined_mapping)

            with st.spinner("Validating and normalising data..."):
                try:
                    normalized_df, warnings = validate_and_normalize(
                        mapped_df,
                        default_timezone=settings.get("default_timezone", "UTC"),
                        logger=logger,
                    )
                    st.session_state.normalized_df = normalized_df
                    st.session_state.validation_warnings = warnings
                    st.session_state.ready_to_enqueue = len(normalized_df) > 0
                except ValueError as exc:
                    st.error(str(exc))
                    st.session_state.ready_to_enqueue = False

            # Show validation results
            st.markdown('<p class="section-title">Validation Results</p>', unsafe_allow_html=True)

            for w in st.session_state.validation_warnings:
                st.warning(w)

            if st.session_state.ready_to_enqueue:
                ndf = st.session_state.normalized_df
                st.success(
                    f"{len(ndf)} valid row(s) ready. "
                    f"{len(raw_df) - len(ndf)} row(s) dropped during validation."
                )

        # --- Data preview panels ---
        col_raw, col_norm = st.columns(2)

        with col_raw:
            st.markdown('<p class="section-title">Raw Uploaded Data</p>', unsafe_allow_html=True)
            st.dataframe(raw_df.head(50), use_container_width=True, hide_index=True)
            if len(raw_df) > 50:
                st.caption(f"Showing first 50 of {len(raw_df)} rows.")

        with col_norm:
            st.markdown('<p class="section-title">Normalised Data (UTC timestamps)</p>', unsafe_allow_html=True)
            if st.session_state.normalized_df is not None and not st.session_state.normalized_df.empty:
                display_df = records_to_df(
                    st.session_state.normalized_df.to_dict("records")
                )
                st.dataframe(display_df.head(50), use_container_width=True, hide_index=True)
                if len(display_df) > 50:
                    st.caption(f"Showing first 50 of {len(display_df)} rows.")
            else:
                st.info("Normalised data will appear here after mapping.")

        # --- Add to queue ---
        st.divider()
        if st.session_state.ready_to_enqueue:
            if st.button("Add to Queue", type="primary", use_container_width=False):
                records = st.session_state.normalized_df.to_dict("records")
                added = scheduler.enqueue(records)
                st.session_state.total_enqueued += added
                st.session_state.notifications.append(
                    f"Added {added} record(s) to the sync queue."
                )
                st.rerun()
        else:
            st.info("Upload a valid file to enable the queue button.")


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

    # Run sync window button
    btn_col, info_col = st.columns([2, 5])

    with btn_col:
        run_clicked = st.button(
            "Run Sync Window",
            type="primary",
            use_container_width=True,
            disabled=scheduler.is_empty(),
        )
    with info_col:
        if scheduler.is_empty():
            st.info("Queue is empty. Upload a file and add records to the queue first.")
        else:
            preview_n = min(scheduler.queue_size(), settings["max_sync_per_window"])
            st.write(
                f"Next window will process up to **{preview_n}** record(s) "
                f"({scheduler.queue_size()} remaining in queue)."
            )

    if run_clicked:
        with st.spinner("Running sync window..."):
            result = engine.run_sync_window()
        if result["status"] == "empty":
            st.warning("Queue was empty when the window ran.")
        else:
            st.success(
                f"Cycle {result['cycle_number']} complete: "
                f"applied {result['applied']}, "
                f"conflicts dropped {result['conflicts']}, "
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
            disabled=scheduler.is_empty(),
        ):
            retries_before = engine.metrics["total_retries"]
            completed = 0
            with st.spinner(f"Running {sim_count} sync cycle(s)..."):
                for _ in range(int(sim_count)):
                    if scheduler.is_empty():
                        break
                    engine.run_sync_window()
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
            disabled=scheduler.is_empty(),
            help="Keeps running sync windows until the queue is completely empty, "
                 "including all retry attempts.",
        ):
            completed_all = 0
            MAX_SAFETY = 500  # prevent infinite loop if retry_limit is very high
            with st.spinner("Running until queue is empty..."):
                while not scheduler.is_empty() and completed_all < MAX_SAFETY:
                    engine.run_sync_window()
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
            f"**{settings.get('scheduling_strategy','fifo').upper()}**)."
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
            cycles_display = pd.DataFrame(ok_cycles)[
                ["cycle_number", "timestamp", "fetched", "applied",
                 "conflicts", "duplicates", "failed", "retried"]
            ].rename(columns={
                "cycle_number": "Cycle",
                "timestamp": "Ran At (UTC)",
                "fetched": "Fetched",
                "applied": "Applied",
                "conflicts": "Conflicts",
                "duplicates": "Duplicates",
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