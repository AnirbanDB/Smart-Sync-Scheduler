"""
conflict_resolver.py

Multi-level conflict resolution for stock updates.

When two or more updates in the same batch target the same (store_id, product),
exactly one must win. Resolution proceeds through three levels:

  Level 1: The update with the later timestamp wins.
  Level 2: On equal timestamps, the update with the lexicographically higher
           update_id wins (provides a deterministic tiebreaker).
  Level 3: On equal update_ids, the last record in input order wins
           (last-write-wins semantics).

Idempotency (requirement 6):
  Duplicate update_ids within a batch are deduplicated before conflict
  resolution runs. True idempotency across sessions is managed by
  SyncEngine using a persistent processed-IDs store.

Out-of-order detection is performed by SyncEngine (not here), because only
the engine has access to the current database state.
"""

from typing import Any, Dict, List, Tuple

from core.logger import get_logger


class ConflictResolver:
    SUPPORTED_POLICIES = ("latest_wins", "multi_level")

    def __init__(self, policy: str = "multi_level", logger=None) -> None:
        if policy not in self.SUPPORTED_POLICIES:
            raise ValueError(
                f"Unknown conflict policy '{policy}'. "
                f"Supported: {', '.join(self.SUPPORTED_POLICIES)}"
            )
        self.policy = policy
        self.logger = logger or get_logger()

    def resolve(
        self, updates: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Resolve conflicts within a single batch.

        Returns
        -------
        (resolved, conflicts_dropped)
        resolved          : deduplicated list, one entry per (store_id, product) pair
        conflicts_dropped : number of records discarded
        """
        if not updates:
            return [], 0

        # Step 1: deduplicate identical update_ids (keep last occurrence)
        uid_seen: Dict[str, Dict[str, Any]] = {}
        for rec in updates:
            uid_seen[str(rec.get("update_id", ""))] = rec
        deduped = list(uid_seen.values())

        # Step 2: apply the configured resolution policy
        resolved, conflicts = self._multi_level(deduped)
        total_dropped = (len(updates) - len(deduped)) + conflicts

        if total_dropped:
            self.logger.info(
                f"Conflict resolution: {len(updates)} in, "
                f"{len(resolved)} resolved, {total_dropped} dropped."
            )
        return resolved, total_dropped

    def check_out_of_order(
        self, incoming: Dict[str, Any], db_entry: Dict[str, Any]
    ) -> bool:
        """
        Return True if the incoming update is older than the current DB entry.

        Both timestamps must be UTC-aware pandas Timestamps for this comparison
        to be valid. If either is missing or incomparable, the check returns
        False (benefit of the doubt).
        """
        rec_ts = incoming.get("timestamp")
        db_ts = db_entry.get("last_updated")
        if rec_ts is None or db_ts is None:
            return False
        try:
            return rec_ts < db_ts
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Resolution strategies
    # ------------------------------------------------------------------

    def _multi_level(
        self, updates: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], int]:
        winners: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for rec in updates:
            key = (str(rec.get("store_id", "")), str(rec.get("product", "")))
            existing = winners.get(key)

            if existing is None:
                winners[key] = rec
                continue

            rec_ts = rec.get("timestamp")
            ex_ts = existing.get("timestamp")

            # Level 1: timestamp comparison
            if rec_ts is not None and ex_ts is not None:
                try:
                    if rec_ts > ex_ts:
                        winners[key] = rec
                        continue
                    if rec_ts < ex_ts:
                        continue
                    # Equal timestamps: fall through to level 2
                except Exception:
                    pass

            # Level 2: update_id comparison (lexicographic)
            rec_uid = str(rec.get("update_id", ""))
            ex_uid = str(existing.get("update_id", ""))
            if rec_uid > ex_uid:
                winners[key] = rec
                continue
            if rec_uid < ex_uid:
                continue

            # Level 3: last-write-wins (current rec comes later in iteration)
            winners[key] = rec

        resolved = list(winners.values())
        conflicts_dropped = len(updates) - len(resolved)
        return resolved, conflicts_dropped