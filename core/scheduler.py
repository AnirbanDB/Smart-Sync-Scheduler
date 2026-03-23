"""
scheduler.py

Queue manager supporting two scheduling strategies:

  fifo     - standard FIFO using collections.deque
  priority - min-heap where stores with fewer total submissions get higher
             priority (prevents starvation, requirement 5)

Server-side receipt stamping:
  When records enter the queue via enqueue(), the scheduler stamps each
  record with received_at = current UTC time. This is the authoritative
  server receipt timestamp used for:
    - Expiry checking in window_checker.py
    - Audit trail in the central database and sync history

  This separates two distinct concepts that were previously conflated:
    timestamp   = when the branch internally recorded this stock level
                  (optional, provided by the branch, used for conflict resolution)
    received_at = when the central server received this update
                  (mandatory, stamped by the server, used for expiry and auditing)
"""

import heapq
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from core.logger import get_logger


class SyncScheduler:
    """FIFO or priority-based queue for pending stock updates."""

    def __init__(
        self,
        max_sync_per_window: int,
        strategy: str = "fifo",
        logger=None,
    ) -> None:
        if max_sync_per_window < 1:
            raise ValueError("max_sync_per_window must be at least 1.")
        if strategy not in ("fifo", "priority"):
            raise ValueError("strategy must be 'fifo' or 'priority'.")

        self.max_sync_per_window = max_sync_per_window
        self.strategy = strategy
        self.logger = logger or get_logger()

        self._fifo: deque = deque()

        # Priority heap entries: (priority, insertion_counter, record_dict)
        self._heap: List[Tuple[int, int, Dict]] = []
        self._heap_counter: int = 0

        # Total records enqueued per store (determines priority score)
        self._store_enqueue_counts: Dict[str, int] = defaultdict(int)

        # Total records dequeued per store (fairness reporting)
        self._store_dequeue_counts: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def enqueue(self, records: List[Dict[str, Any]]) -> int:
        """
        Add records to the queue and stamp each one with received_at.

        received_at is the authoritative server receipt timestamp. It is
        set here — at the boundary between the outside world and the
        processing pipeline — so it accurately reflects when the central
        server accepted the data, regardless of what the branch reported
        internally.
        """
        added = 0
        server_time = datetime.now(tz=timezone.utc)

        for rec in records:
            # Stamp with server receipt time. Never overwrite an existing
            # received_at (supports re-enqueue on retry without losing the
            # original receipt time).
            if "received_at" not in rec:
                rec["received_at"] = server_time

            if self.strategy == "priority":
                store = str(rec.get("store_id", ""))
                priority = self._store_enqueue_counts[store]
                heapq.heappush(self._heap, (priority, self._heap_counter, rec))
                self._heap_counter += 1
                self._store_enqueue_counts[store] += 1
            else:
                self._fifo.append(rec)
            added += 1

        self.logger.info(
            f"Enqueued {added} record(s). Queue size: {self.queue_size()}."
        )
        return added

    def dequeue_batch(self) -> List[Dict[str, Any]]:
        """Remove and return up to max_sync_per_window records (FIFO order)."""
        batch: List[Dict[str, Any]] = []

        if self.strategy == "priority":
            while self._heap and len(batch) < self.max_sync_per_window:
                _, _, rec = heapq.heappop(self._heap)
                batch.append(rec)
        else:
            while self._fifo and len(batch) < self.max_sync_per_window:
                batch.append(self._fifo.popleft())

        for rec in batch:
            store = str(rec.get("store_id", ""))
            self._store_dequeue_counts[store] += 1

        return batch

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    def peek(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return up to n records without removing them."""
        if self.strategy == "priority":
            return [rec for _, _, rec in sorted(self._heap)[:n]]
        return [self._fifo[i] for i in range(min(n, len(self._fifo)))]

    def get_all_pending(self) -> List[Dict[str, Any]]:
        """Return all pending records as a list (non-destructive)."""
        if self.strategy == "priority":
            return [rec for _, _, rec in sorted(self._heap)]
        return list(self._fifo)

    def queue_size(self) -> int:
        if self.strategy == "priority":
            return len(self._heap)
        return len(self._fifo)

    def is_empty(self) -> bool:
        return self.queue_size() == 0

    def store_counts(self) -> Dict[str, int]:
        """Records dequeued per store (fairness reporting)."""
        return dict(self._store_dequeue_counts)

    def clear(self) -> None:
        self._fifo.clear()
        self._heap.clear()
        self._store_enqueue_counts.clear()
        self._store_dequeue_counts.clear()