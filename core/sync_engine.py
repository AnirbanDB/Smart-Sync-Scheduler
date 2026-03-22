"""
sync_engine.py

Orchestrates sync windows and maintains the central stock database.

Key guarantees:

  Atomic batch processing (requirement 18):
    Updates are first written to a deep copy of the database. If all writes
    succeed, the copy replaces the live database. Any exception triggers a
    rollback to the original snapshot.

  Idempotency persistence (requirement 19):
    Processed update_ids are saved to a JSON file after every successful
    batch. On startup, this file is loaded so duplicate processing is
    prevented even across application restarts.

  Metrics (requirement 20):
    The engine tracks processed, failed, retried, conflict-dropped, and
    duplicate-skipped counts, plus a queue-size history for charting.

  Out-of-order protection:
    Before committing an update to the temp database, the engine checks
    whether the incoming timestamp is older than the currently stored value.
    Stale records are treated as failures and sent through the retry handler.
"""

import copy
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Set, Tuple

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

        # Persistent processed ID set (loaded from disk)
        self._processed_ids: Set[str] = self._load_processed_ids()

        # Audit log of every record that has moved through the engine
        self.processed_records: List[Dict[str, Any]] = []

        # Per-cycle summary
        self.sync_cycles: List[Dict[str, Any]] = []

        # Metrics (requirement 20)
        self.metrics: Dict[str, Any] = {
            "total_processed": 0,
            "total_failed": 0,
            "total_retries": 0,
            "total_conflicts": 0,
            "total_duplicates_skipped": 0,
            "queue_size_history": [],
        }

    # ------------------------------------------------------------------
    # Main sync window
    # ------------------------------------------------------------------

    def run_sync_window(self) -> Dict[str, Any]:
        """
        Pull one batch from the queue and apply it atomically.

        Returns a result dict with cycle statistics.
        """
        cycle_num = len(self.sync_cycles) + 1
        run_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # Record queue size before dequeue
        self.metrics["queue_size_history"].append(
            {"cycle": cycle_num, "size": self.scheduler.queue_size(), "timestamp": run_ts}
        )

        if self.scheduler.is_empty():
            result = self._empty_cycle_result(cycle_num, run_ts)
            self.sync_cycles.append(result)
            return result

        batch = self.scheduler.dequeue_batch()
        self.logger.info(f"Cycle {cycle_num}: dequeued {len(batch)} records.")

        # Stage 1: Idempotency filter
        fresh_records, duplicate_ids = self._filter_duplicates(batch)
        duplicates_skipped = len(duplicate_ids)
        self.metrics["total_duplicates_skipped"] += duplicates_skipped

        # Stage 2: Conflict resolution (within-batch)
        resolved_records, conflicts_dropped = self.resolver.resolve(fresh_records)
        self.metrics["total_conflicts"] += conflicts_dropped

        # Build ID sets for fast status lookup when archiving
        resolved_ids = {str(r.get("update_id", "")) for r in resolved_records}
        conflict_dropped_ids = (
            {str(r.get("update_id", "")) for r in fresh_records} - resolved_ids
        )

        # Stage 3: Atomic apply
        applied_records, failed_records, failed_with_reasons = self._atomic_apply(
            resolved_records, cycle_num, run_ts
        )
        applied_ids = {str(r.get("update_id", "")) for r in applied_records}
        failed_ids = {str(r.get("update_id", "")) for r in failed_records}

        # Update persistent processed IDs
        for rec in applied_records:
            uid = str(rec.get("update_id", ""))
            if uid:
                self._processed_ids.add(uid)
        self._save_processed_ids()

        # Update metrics
        self.metrics["total_processed"] += len(applied_records)
        self.metrics["total_failed"] += len(failed_records)

        # Archive every record with its final status
        self._archive_batch(
            batch=batch,
            cycle_num=cycle_num,
            applied_ids=applied_ids,
            failed_ids=failed_ids,
            duplicate_ids=duplicate_ids,
            conflict_dropped_ids=conflict_dropped_ids,
        )

        # Retry failed records
        requeued = self._handle_retries(failed_with_reasons)
        self.metrics["total_retries"] += requeued

        result: Dict[str, Any] = {
            "status": "ok",
            "cycle_number": cycle_num,
            "timestamp": run_ts,
            "fetched": len(batch),
            "applied": len(applied_records),
            "conflicts": conflicts_dropped,
            "duplicates": duplicates_skipped,
            "failed": len(failed_records),
            "retried": requeued,
            "applied_records": applied_records,
        }
        self.sync_cycles.append(result)
        self.logger.info(
            f"Cycle {cycle_num} done: applied={len(applied_records)}, "
            f"conflicts={conflicts_dropped}, failed={len(failed_records)}, "
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
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Apply records to a temporary copy of the database.
        On success: replace live DB with the temp copy.
        On exception: discard temp copy, preserve original state.
        """
        db_snapshot = copy.deepcopy(self._database)
        temp_db = copy.deepcopy(self._database)
        applied: List[Dict[str, Any]] = []
        # Each entry is (record, failure_reason_string)
        failed_with_reasons: List[Tuple[Dict[str, Any], str]] = []

        try:
            for rec in records:
                key = (str(rec.get("store_id", "")), str(rec.get("product", "")))
                db_entry = temp_db.get(key)

                # Out-of-order check: reject stale data
                if db_entry and self.resolver.check_out_of_order(rec, db_entry):
                    db_ts = db_entry.get("last_updated", "")
                    rec_ts = rec.get("timestamp", "")
                    db_ts_str = db_ts.isoformat(timespec="seconds") if hasattr(db_ts, "isoformat") else str(db_ts)
                    rec_ts_str = rec_ts.isoformat(timespec="seconds") if hasattr(rec_ts, "isoformat") else str(rec_ts)
                    reason = f"Stale update (record: {rec_ts_str} < db: {db_ts_str})"
                    self.logger.warning(
                        f"Out-of-order update ignored: update_id={rec.get('update_id')}. "
                        f"{reason}"
                    )
                    failed_with_reasons.append((rec, reason))
                    continue

                temp_db[key] = {
                    "store_id": rec.get("store_id"),
                    "product": rec.get("product"),
                    "quantity": rec.get("quantity"),
                    "last_updated": rec.get("timestamp"),
                    "update_id": rec.get("update_id"),
                    "synced_at": run_ts,
                    "sync_cycle": cycle_num,
                }
                applied.append(rec)

            # Commit: replace live database with temp copy
            self._database = temp_db
            self.logger.info(f"Atomic commit: {len(applied)} record(s) written.")

        except Exception as exc:
            # Rollback: restore snapshot
            self._database = db_snapshot
            self.logger.error(f"Atomic batch failed, rolling back: {exc}")
            failed_with_reasons = [(r, f"Processing error: {str(exc)[:60]}") for r in records]
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

    def _handle_retries(self, failed_with_reasons: List[Tuple[Dict[str, Any], str]]) -> int:
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
                # Build a final DLQ reason that includes both root cause and retry count
                self.retry_handler.send_to_dead_letter(rec, reason)
        return requeued

    # ------------------------------------------------------------------
    # Archiving
    # ------------------------------------------------------------------

    def _archive_batch(
        self,
        batch: List[Dict[str, Any]],
        cycle_num: int,
        applied_ids: Set[str],
        failed_ids: Set[str],
        duplicate_ids: Set[str],
        conflict_dropped_ids: Set[str],
    ) -> None:
        for rec in batch:
            uid = str(rec.get("update_id", ""))
            archived = dict(rec)
            archived["sync_cycle"] = cycle_num

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
        Clear in-session state. Does NOT clear the idempotency store on disk,
        so processed IDs remain durable across session resets (requirement 19).
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
            "queue_size_history": [],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_cycle_result(cycle_num: int, run_ts: str) -> Dict[str, Any]:
        return {
            "status": "empty",
            "cycle_number": cycle_num,
            "timestamp": run_ts,
            "fetched": 0,
            "applied": 0,
            "conflicts": 0,
            "duplicates": 0,
            "failed": 0,
            "retried": 0,
            "applied_records": [],
        }


def _resolve(path: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, path)