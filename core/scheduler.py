"""
scheduler.py

Queue manager supporting two scheduling strategies:

  fifo     - standard FIFO using collections.deque
  priority - min-heap where records from underserved stores get higher priority
             (fair scheduling to prevent starvation, requirement 5)

Fair scheduling detail:
  Priority value = number of records from that store already enqueued in this
  session. Stores with fewer total submissions therefore sit higher in the heap
  and are served first. This is a static score assigned at enqueue time; no
  re-ordering happens later, which keeps heap operations O(log n).
"""

import heapq
from collections import defaultdict, deque
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
        added = 0
        for rec in records:
            if self.strategy == "priority":
                store = str(rec.get("store_id", ""))
                # Priority = total enqueues so far for this store before this record.
                # Lower value = higher heap priority (min-heap).
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
        if self.strategy == "priority":
            return [rec for _, _, rec in sorted(self._heap)[:n]]
        return [self._fifo[i] for i in range(min(n, len(self._fifo)))]

    def get_all_pending(self) -> List[Dict[str, Any]]:
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
        """Records dequeued per store so far (for fairness reporting)."""
        return dict(self._store_dequeue_counts)

    def clear(self) -> None:
        self._fifo.clear()
        self._heap.clear()
        self._store_enqueue_counts.clear()
        self._store_dequeue_counts.clear()