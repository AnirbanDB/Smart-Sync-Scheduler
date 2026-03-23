"""
sync_engine.py

Orchestrates sync windows and maintains the central stock database.

Key guarantees:

  Atomic batch processing (requirement 18):
    Updates are written to a deep copy of the database. On success the copy
    replaces the live database. Any exception triggers a full rollback.

  Idempotency persistence (requirement 19):
    Processed update_ids are saved to a JSON file after every successful
    batch so duplicate processing is prevented even across restarts.

  Metrics (requirement 20):
    Tracks processed, failed, retried, conflict-dropped, duplicate-skipped,
    and expired counts, plus a queue-size history for charting.

  Out-of-order protection:
    Before committing, the engine checks whether the incoming timestamp is
    older than the currently stored value for that store/product pair.
    Stale records are failed and retried up to the configured limit.

  Sync window integration:
    run_sync_window accepts a window_label (recorded in the audit log and
    central database) and a WindowChecker instance. The WindowChecker uses
    received_at — the server-stamped queue entry time — to identify records
    that have waited too long and should be expired to the DLQ.

  Two timestamp fields:
    timestamp   = branch-provided recording time (optional, used for
                  conflict resolution to determine which stock level is
                  most recent)
    received_at = server-stamped queue entry time (always present after
                  scheduler.enqueue(), used for expiry and audit)
"""

import copy
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from core.conflict_resolver import ConflictResolver
from core.logger import get_logger
from core.retry_handler import RetryHandler
from core.scheduler import SyncScheduler


class SyncEngine:
    def __init__(
        self,
        scheduler: SyncScheduler,
        resolver: ConflictResolver,
        retry_handler: RetryHandler,
        idempotency_store_path: str = "data/processed_ids.json",
        logger=None,
    ) -> None:
        self.scheduler = scheduler
        self.resolver = resolver
        self.retry_handler = retry_handler
        self.idempotency_store_path = idempotency_store_path
        self.logger = logger or get_logger()

        # Central stock database: (store_id, product) -> record dict
        self._database: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # Persistent processed ID set loaded from disk on startup
        self._processed_ids: Set[str] = self._load_processed_ids()

        # Audit log of every record that has moved through the engine
        self.processed_records: List[Dict[str, Any]] = []

        # Per-cycle summary entries
        self.sync_cycles: List[Dict[str, Any]] = []

        # Live metrics
        self.metrics: Dict[str, Any] = {
            "total_processed": 0,
            "total_failed": 0,
            "total_retries": 0,
            "total_conflicts": 0,
            "total_duplicates_skipped": 0,
            "total_expired": 0,
            "queue_size_history": [],
        }

    # ------------------------------------------------------------------
    # Main sync window
    # ------------------------------------------------------------------

    def run_sync_window(
        self,
        window_label: str = "Manual",
        window_checker=None,
    ) -> Dict[str, Any]:
        """
        Pull one batch from the queue and apply it atomically.

        Parameters
        ----------
        window_label   : Name of the sync window triggering this run.
                         Written into every processed record and DB entry.
        window_checker : Optional WindowChecker instance. When provided,
                         records whose received_at is older than
                         update_expiry_hours are moved to the dead letter
                         queue before any processing occurs.

        Returns a result dict with full cycle statistics.
        """
        cycle_num = len(self.sync_cycles) + 1
        run_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

        self.metrics["queue_size_history"].append(
            {"cycle": cycle_num, "size": self.scheduler.queue_size(), "timestamp": run_ts}
        )

        if self.scheduler.is_empty():
            result = self._empty_cycle_result(cycle_num, run_ts, window_label)
            self.sync_cycles.append(result)
            return result

        batch = self.scheduler.dequeue_batch()
        self.logger.info(
            f"Cycle {cycle_num} [{window_label}]: dequeued {len(batch)} record(s)."
        )

        # ------------------------------------------------------------------
        # Stage 0: Expiry check
        # Records that have been queued too long (based on received_at, the
        # server-stamped receipt time) go directly to the dead letter queue.
        # They are not retried because the underlying data is too stale.
        # ------------------------------------------------------------------
        expired_count = 0
        if window_checker is not None:
            live_batch = []
            for rec in batch:
                if window_checker.is_update_expired(rec.get("received_at")):
                    reason = (
                        f"Update expired: queued for longer than "
                        f"{window_checker.expiry_hours:.0f}h without being processed."
                    )
                    self.retry_handler.send_to_dead_letter(rec, reason)
                    expired_count += 1
                    self.logger.warning(
                        f"Expired record sent to DLQ: update_id={rec.get('update_id')}, "
                        f"received_at={rec.get('received_at')}"
                    )
                else:
                    live_batch.append(rec)
            batch = live_batch

        # Stage 1: Idempotency filter
        fresh_records, duplicate_ids = self._filter_duplicates(batch)
        duplicates_skipped = len(duplicate_ids)
        self.metrics["total_duplicates_skipped"] += duplicates_skipped

        # Stage 2: Conflict resolution (within-batch)
        resolved_records, conflicts_dropped = self.resolver.resolve(fresh_records)
        self.metrics["total_conflicts"] += conflicts_dropped

        resolved_ids = {str(r.get("update_id", "")) for r in resolved_records}
        conflict_dropped_ids = (
            {str(r.get("update_id", "")) for r in fresh_records} - resolved_ids
        )

        # Stage 3: Atomic apply
        applied_records, failed_records, failed_with_reasons = self._atomic_apply(
            resolved_records, cycle_num, run_ts, window_label
        )
        applied_ids = {str(r.get("update_id", "")) for r in applied_records}
        failed_ids  = {str(r.get("update_id", "")) for r in failed_records}

        for rec in applied_records:
            uid = str(rec.get("update_id", ""))
            if uid:
                self._processed_ids.add(uid)
        self._save_processed_ids()

        self.metrics["total_processed"] += len(applied_records)
        self.metrics["total_failed"]    += len(failed_records)
        self.metrics["total_expired"]   += expired_count

        self._archive_batch(
            batch=batch,
            cycle_num=cycle_num,
            window_label=window_label,
            applied_ids=applied_ids,
            failed_ids=failed_ids,
            duplicate_ids=duplicate_ids,
            conflict_dropped_ids=conflict_dropped_ids,
        )

        requeued = self._handle_retries(failed_with_reasons)
        self.metrics["total_retries"] += requeued

        result: Dict[str, Any] = {
            "status": "ok",
            "cycle_number": cycle_num,
            "timestamp": run_ts,
            "window_label": window_label,
            "fetched": len(batch),
            "applied": len(applied_records),
            "conflicts": conflicts_dropped,
            "duplicates": duplicates_skipped,
            "expired": expired_count,
            "failed": len(failed_records),
            "retried": requeued,
            "applied_records": applied_records,
        }
        self.sync_cycles.append(result)
        self.logger.info(
            f"Cycle {cycle_num} [{window_label}] done: "
            f"applied={len(applied_records)}, conflicts={conflicts_dropped}, "
            f"expired={expired_count}, failed={len(failed_records)}, "
            f"retried={requeued}."
        )
        return result

    # ------------------------------------------------------------------
    # Atomic apply (requirement 18)
    # ------------------------------------------------------------------

    def _atomic_apply(
        self,
        records: List[Dict[str, Any]],
        cycle_num: int,
        run_ts: str,
        window_label: str = "Manual",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List]:
        """
        Write records to a temporary copy of the database.
        On success: replace live DB. On any exception: rollback to snapshot.

        Conflict resolution within the central database:
          The engine uses the record's timestamp field (branch recording time)
          when present for out-of-order detection, because that field represents
          the actual recency of the stock information. If timestamp is absent,
          received_at is used as a fallback so the system still rejects stale
          re-submissions of the same record.
        """
        db_snapshot = copy.deepcopy(self._database)
        temp_db     = copy.deepcopy(self._database)
        applied: List[Dict[str, Any]] = []
        failed_with_reasons: List[Tuple[Dict[str, Any], str]] = []

        try:
            for rec in records:
                key      = (str(rec.get("store_id", "")), str(rec.get("product", "")))
                db_entry = temp_db.get(key)

                # Out-of-order check:
                # Prefer timestamp (branch recording time) for comparison.
                # Fall back to received_at if timestamp is absent.
                if db_entry:
                    incoming_ts = rec.get("timestamp") or rec.get("received_at")
                    if incoming_ts is not None and self.resolver.check_out_of_order(
                        {"timestamp": incoming_ts}, db_entry
                    ):
                        db_ts  = db_entry.get("last_updated", "")
                        rec_ts = incoming_ts
                        db_ts_str  = db_ts.isoformat(timespec="seconds")  if hasattr(db_ts,  "isoformat") else str(db_ts)
                        rec_ts_str = rec_ts.isoformat(timespec="seconds") if hasattr(rec_ts, "isoformat") else str(rec_ts)
                        reason = (
                            f"Stale update (record: {rec_ts_str} < db: {db_ts_str})"
                        )
                        self.logger.warning(
                            f"Out-of-order update ignored: "
                            f"update_id={rec.get('update_id')}. {reason}"
                        )
                        failed_with_reasons.append((rec, reason))
                        continue

                # Store both the branch timestamp and the server receipt time
                # so the full audit trail is preserved.
                temp_db[key] = {
                    "store_id":    rec.get("store_id"),
                    "product":     rec.get("product"),
                    "quantity":    rec.get("quantity"),
                    "timestamp":   rec.get("timestamp"),
                    "received_at": rec.get("received_at"),
                    "last_updated": rec.get("timestamp") or rec.get("received_at"),
                    "update_id":   rec.get("update_id"),
                    "synced_at":   run_ts,
                    "sync_cycle":  cycle_num,
                    "sync_window": window_label,
                }
                applied.append(rec)

            self._database = temp_db
            self.logger.info(
                f"Atomic commit: {len(applied)} record(s) written to database."
            )

        except Exception as exc:
            self._database = db_snapshot
            self.logger.error(f"Atomic batch failed, rolling back: {exc}")
            failed_with_reasons = [
                (r, f"Processing error: {str(exc)[:60]}") for r in records
            ]
            applied = []

        failed_records = [r for r, _ in failed_with_reasons]
        return applied, failed_records, failed_with_reasons

    # ------------------------------------------------------------------
    # Idempotency (requirement 19)
    # ------------------------------------------------------------------

    def _filter_duplicates(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Set[str]]:
        fresh: List[Dict[str, Any]] = []
        duplicate_ids: Set[str] = set()
        for rec in batch:
            uid = str(rec.get("update_id", ""))
            if uid in self._processed_ids:
                duplicate_ids.add(uid)
                self.logger.info(f"Duplicate skipped: update_id={uid}")
            else:
                fresh.append(rec)
        return fresh, duplicate_ids

    def _load_processed_ids(self) -> Set[str]:
        path = _resolve(self.idempotency_store_path)
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                ids = set(data.get("processed_ids", []))
                self.logger.info(
                    f"Loaded {len(ids)} processed ID(s) from idempotency store."
                )
                return ids
        except Exception as exc:
            self.logger.warning(f"Could not load idempotency store: {exc}")
        return set()

    def _save_processed_ids(self) -> None:
        path = _resolve(self.idempotency_store_path)
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"processed_ids": list(self._processed_ids)}, fh)
        except Exception as exc:
            self.logger.warning(f"Could not save idempotency store: {exc}")

    # ------------------------------------------------------------------
    # Retry handling
    # ------------------------------------------------------------------

    def _handle_retries(
        self, failed_with_reasons: List[Tuple[Dict[str, Any], str]]
    ) -> int:
        requeued = 0
        for rec, reason in failed_with_reasons:
            uid = str(rec.get("update_id", ""))
            if self.retry_handler.should_retry(uid):
                count = self.retry_handler.record_retry(uid, reason)
                self.scheduler.enqueue([rec])
                requeued += 1
                self.logger.warning(
                    f"Requeued update_id={uid} for retry #{count}. Reason: {reason}"
                )
            else:
                self.retry_handler.send_to_dead_letter(rec, reason)
        return requeued

    # ------------------------------------------------------------------
    # Archiving
    # ------------------------------------------------------------------

    def _archive_batch(
        self,
        batch: List[Dict[str, Any]],
        cycle_num: int,
        window_label: str,
        applied_ids: Set[str],
        failed_ids: Set[str],
        duplicate_ids: Set[str],
        conflict_dropped_ids: Set[str],
    ) -> None:
        for rec in batch:
            uid = str(rec.get("update_id", ""))
            archived = dict(rec)
            archived["sync_cycle"]  = cycle_num
            archived["sync_window"] = window_label

            if uid in applied_ids:
                archived["sync_status"] = "applied"
            elif uid in failed_ids:
                archived["sync_status"] = "failed"
            elif uid in duplicate_ids:
                archived["sync_status"] = "duplicate_skipped"
            elif uid in conflict_dropped_ids:
                archived["sync_status"] = "conflict_dropped"
            else:
                archived["sync_status"] = "unknown"

            self.processed_records.append(archived)

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    def get_database_records(self) -> List[Dict[str, Any]]:
        return list(self._database.values())

    def total_processed(self) -> int:
        return self.metrics["total_processed"]

    def total_conflicts(self) -> int:
        return self.metrics["total_conflicts"]

    # ------------------------------------------------------------------
    # Session reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Clear all in-session state. The idempotency store on disk is
        preserved so processed IDs remain durable across manual resets.
        """
        self._database.clear()
        self.processed_records.clear()
        self.sync_cycles.clear()
        self.scheduler.clear()
        self.retry_handler.clear()
        self.metrics = {
            "total_processed": 0,
            "total_failed": 0,
            "total_retries": 0,
            "total_conflicts": 0,
            "total_duplicates_skipped": 0,
            "total_expired": 0,
            "queue_size_history": [],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_cycle_result(
        cycle_num: int,
        run_ts: str,
        window_label: str = "Manual",
    ) -> Dict[str, Any]:
        return {
            "status": "empty",
            "cycle_number": cycle_num,
            "timestamp": run_ts,
            "window_label": window_label,
            "fetched": 0,
            "applied": 0,
            "conflicts": 0,
            "duplicates": 0,
            "expired": 0,
            "failed": 0,
            "retried": 0,
            "applied_records": [],
        }


def _resolve(path: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, path)