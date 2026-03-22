"""
retry_handler.py

Manages retry logic for updates that fail during sync window processing.

An update that fails is retried up to retry_limit times. After exhausting
all retries, the record is moved to the dead letter queue (DLQ) where it
can be inspected in the UI.

This is separate from conflict resolution: a conflict drop is a deliberate
decision, not a failure. A failure is an unexpected error (e.g. out-of-order
data rejected by the database state).
"""

from typing import Any, Dict, List

from core.logger import get_logger


class RetryHandler:
    """
    Tracks retry counts per update_id and maintains the dead letter queue.
    """

    def __init__(self, retry_limit: int = 3, logger=None) -> None:
        if retry_limit < 0:
            raise ValueError("retry_limit must be >= 0.")
        self.retry_limit = retry_limit
        self.logger = logger or get_logger()

        # update_id -> number of times retried so far
        self._retry_counts: Dict[str, int] = {}

        # update_id -> last known failure reason (updated on every retry)
        self._last_reasons: Dict[str, str] = {}

        # Records that have exceeded the retry limit
        self._dead_letter: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Retry control
    # ------------------------------------------------------------------

    def should_retry(self, update_id: str) -> bool:
        """Return True if this update has not yet hit the retry ceiling."""
        return self._retry_counts.get(str(update_id), 0) < self.retry_limit

    def record_retry(self, update_id: str, reason: str = "") -> int:
        """
        Increment the retry counter, store the failure reason, and return
        the new count. The reason is kept so the DLQ shows why the record
        kept failing, not just that retries were exhausted.
        """
        uid = str(update_id)
        new_count = self._retry_counts.get(uid, 0) + 1
        self._retry_counts[uid] = new_count
        if reason:
            self._last_reasons[uid] = reason
        self.logger.warning(
            f"Retry #{new_count}/{self.retry_limit} scheduled for update_id={uid}."
        )
        return new_count

    def send_to_dead_letter(self, record: Dict[str, Any], reason: str) -> None:
        """Move a record to the dead letter queue after all retries are exhausted."""
        entry = dict(record)
        uid = str(record.get("update_id", "unknown"))
        entry["failure_reason"] = reason
        entry["retry_count"] = self._retry_counts.get(uid, 0)
        self._dead_letter.append(entry)
        self.logger.error(
            f"update_id={uid} moved to dead letter queue after "
            f"{entry['retry_count']} retries. Reason: {reason}"
        )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get_dead_letter_queue(self) -> List[Dict[str, Any]]:
        return list(self._dead_letter)

    def get_retry_count(self, update_id: str) -> int:
        return self._retry_counts.get(str(update_id), 0)

    def total_retries(self) -> int:
        return sum(self._retry_counts.values())

    def clear(self) -> None:
        self._retry_counts.clear()
        self._last_reasons.clear()
        self._dead_letter.clear()