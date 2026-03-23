"""
window_checker.py

Pure time-window logic for the sync schedule system.

Responsibilities:
  - Parse window definitions from the settings dict.
  - Determine whether the current moment falls inside any configured window.
  - Calculate time remaining until the current window closes.
  - Calculate time until the next window opens.
  - Determine whether a queued update has exceeded its queue wait expiry.
    Expiry is checked against received_at (server receipt time), not the
    branch-provided timestamp field.
  - Validate new window entries before saving.
  - Write an updated schedule back to settings.json.

This module has no Streamlit dependency. All time comparisons use
timezone-aware datetimes. The standard library zoneinfo module is used
for timezone handling (available in Python 3.9 and above), so no third-party
timezone library is required.

Design note on daily repetition:
  Windows are defined as HH:MM - HH:MM and repeat every day automatically.
  The head office configures them once; all branches follow the same schedule
  indefinitely. Changes take effect from the next occurrence of the window.
"""

import json
import os
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from core.logger import get_logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class SyncWindow:
    """
    One named time window within a 24-hour day.

    Parameters
    ----------
    label : Human-readable name, e.g. "Morning Sync".
    start : Window open time (naive, interpreted in the schedule timezone).
    end   : Window close time (naive, interpreted in the schedule timezone).
    """

    def __init__(self, label: str, start: time, end: time) -> None:
        self.label = label
        self.start = start
        self.end = end

    def to_dict(self) -> Dict[str, str]:
        return {
            "label": self.label,
            "start": self.start.strftime("%H:%M"),
            "end":   self.end.strftime("%H:%M"),
        }

    def __repr__(self) -> str:
        return (
            f"SyncWindow(label={self.label!r}, "
            f"start={self.start.strftime('%H:%M')}, "
            f"end={self.end.strftime('%H:%M')})"
        )


class WindowStatus:
    """
    Snapshot of the current window state, computed at a single point in time.

    Attributes
    ----------
    is_open              : True if the current time is inside an active window.
    current_window       : The active SyncWindow, or None if the server is closed.
    next_window          : The SyncWindow that will open next, or None if none defined.
    seconds_until_change : Seconds until the next state change (open->closed or
                           closed->open). None if no future window can be found.
    now_local            : The current timezone-aware datetime used for this check.
    """

    def __init__(
        self,
        is_open: bool,
        current_window: Optional[SyncWindow],
        next_window: Optional[SyncWindow],
        seconds_until_change: Optional[float],
        now_local: datetime,
    ) -> None:
        self.is_open = is_open
        self.current_window = current_window
        self.next_window = next_window
        self.seconds_until_change = seconds_until_change
        self.now_local = now_local

    def time_until_change_str(self) -> str:
        """Return a human-readable countdown, e.g. '2h 13m' or '45m'."""
        if self.seconds_until_change is None:
            return "unknown"
        total = int(self.seconds_until_change)
        if total <= 0:
            return "now"
        hours, remainder = divmod(total, 3600)
        minutes = remainder // 60
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"


# ---------------------------------------------------------------------------
# Core checker
# ---------------------------------------------------------------------------

class WindowChecker:
    """
    Evaluates configured sync windows against the live clock.

    Parameters
    ----------
    schedule_config : the "sync_schedule" sub-dict from settings.json.
    logger          : optional logger instance.
    """

    def __init__(self, schedule_config: Dict[str, Any], logger=None) -> None:
        self.logger = logger or get_logger()
        self.enforce      = schedule_config.get("enforce_window", True)
        self.expiry_hours = float(schedule_config.get("update_expiry_hours", 48))

        tz_name = schedule_config.get("timezone", "UTC")
        try:
            self.tz = ZoneInfo(tz_name)
        except (ZoneInfoNotFoundError, KeyError):
            self.logger.warning(
                f"Unknown timezone '{tz_name}'. Falling back to UTC."
            )
            self.tz = ZoneInfo("UTC")

        self.windows: List[SyncWindow] = _parse_windows(
            schedule_config.get("windows", [])
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_status(self) -> WindowStatus:
        """Return a WindowStatus snapshot based on the live clock."""
        now = datetime.now(tz=self.tz)
        now_time = now.time().replace(second=0, microsecond=0)

        active = _find_active_window(now_time, self.windows)
        next_win, next_dt = _find_next_window(now, active, self.windows)

        if active is not None:
            close_dt = datetime.combine(now.date(), active.end).replace(tzinfo=self.tz)
            if close_dt <= now:
                close_dt += timedelta(days=1)
            secs = max(0.0, (close_dt - now).total_seconds())
            return WindowStatus(
                is_open=True,
                current_window=active,
                next_window=next_win,
                seconds_until_change=secs,
                now_local=now,
            )

        secs_until = (
            max(0.0, (next_dt - now).total_seconds())
            if next_dt is not None else None
        )
        return WindowStatus(
            is_open=False,
            current_window=None,
            next_window=next_win,
            seconds_until_change=secs_until,
            now_local=now,
        )

    def is_update_expired(self, received_at) -> bool:
        """
        Return True if the record has been sitting in the queue longer than
        update_expiry_hours without being processed.

        This method checks received_at — the server-stamped time when the
        record entered the queue — not the branch-provided timestamp field.
        This is the correct design because:

          - The branch timestamp reflects when the branch internally recorded
            the stock level, which may legitimately be hours old.
          - The expiry concern is about queue age: how long has this update
            been waiting on the server without being applied? If it has
            waited too long, the stock level it reports may have been
            superseded by real-world events even if a newer update was not
            submitted.

        Parameters
        ----------
        received_at : The server receipt timestamp stored on the record.
                      Accepts a pandas Timestamp or datetime. Returns False
                      (benefit of the doubt) if the value is missing or
                      cannot be compared.
        """
        if received_at is None or self.expiry_hours <= 0:
            return False
        try:
            now_utc = datetime.now(tz=ZoneInfo("UTC"))
            cutoff  = now_utc - timedelta(hours=self.expiry_hours)
            ts = received_at
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=ZoneInfo("UTC"))
            return ts < cutoff
        except Exception:
            return False

    def processing_allowed(self, override: bool = False) -> Tuple[bool, str]:
        """
        Return (allowed, reason).

        Enforcement off or override active -> always allowed.
        Otherwise checks the live clock against configured windows.
        """
        if not self.enforce:
            return True, "Window enforcement is disabled. Processing allowed at any time."
        if override:
            return True, "Emergency override is active. Processing bypasses the schedule."

        status = self.get_status()
        if status.is_open:
            return True, (
                f"Window open: {status.current_window.label}. "
                f"Closes in {status.time_until_change_str()}."
            )
        if status.next_window is not None:
            return False, (
                f"Server is outside sync window. "
                f"Next: {status.next_window.label}. "
                f"Opens in {status.time_until_change_str()}."
            )
        return False, "No sync windows are configured. Processing is blocked."


# ---------------------------------------------------------------------------
# Internal helpers (module-level, not methods, so they are easy to test)
# ---------------------------------------------------------------------------

def _parse_windows(raw: List[Dict[str, str]]) -> List[SyncWindow]:
    windows = []
    for entry in raw:
        try:
            start = datetime.strptime(entry["start"], "%H:%M").time()
            end   = datetime.strptime(entry["end"],   "%H:%M").time()
            label = entry.get("label", f"{entry['start']}-{entry['end']}")
            if start >= end:
                continue
            windows.append(SyncWindow(label=label, start=start, end=end))
        except Exception:
            continue
    return windows


def _find_active_window(
    now_time: time,
    windows: List[SyncWindow],
) -> Optional[SyncWindow]:
    for win in windows:
        if win.start <= now_time <= win.end:
            return win
    return None


def _find_next_window(
    now: datetime,
    active: Optional[SyncWindow],
    windows: List[SyncWindow],
) -> Tuple[Optional[SyncWindow], Optional[datetime]]:
    """
    Find the next window that will open after the current moment.
    Looks up to 24 hours ahead to handle overnight gaps.
    """
    if not windows:
        return None, None

    tz = now.tzinfo
    sorted_wins = sorted(windows, key=lambda w: w.start)
    now_time = now.time().replace(second=0, microsecond=0)

    for win in sorted_wins:
        if win is active:
            continue
        if win.start > now_time:
            next_dt = datetime.combine(now.date(), win.start).replace(tzinfo=tz)
            return win, next_dt

    # Wrap around to tomorrow's first window
    first = sorted_wins[0]
    next_dt = datetime.combine(
        now.date() + timedelta(days=1), first.start
    ).replace(tzinfo=tz)
    return first, next_dt


# ---------------------------------------------------------------------------
# Schedule persistence
# ---------------------------------------------------------------------------

def load_schedule_config(
    settings_path: str = "config/settings.json",
) -> Dict[str, Any]:
    """Read the sync_schedule block from settings.json."""
    abs_path = _resolve(settings_path)
    try:
        with open(abs_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("sync_schedule", _default_schedule())
    except Exception:
        return _default_schedule()


def save_schedule_config(
    schedule: Dict[str, Any],
    settings_path: str = "config/settings.json",
) -> bool:
    """
    Write an updated sync_schedule block back into settings.json.
    Returns True on success, False on failure.
    """
    abs_path = _resolve(settings_path)
    try:
        with open(abs_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        data["sync_schedule"] = schedule
        with open(abs_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4)
        return True
    except Exception:
        return False


def validate_window_entry(
    label: str, start: str, end: str
) -> Tuple[bool, str]:
    """
    Validate a new window entry before adding it.
    Returns (valid, error_message). error_message is empty on success.
    """
    if not label.strip():
        return False, "Window label cannot be empty."
    try:
        s = datetime.strptime(start.strip(), "%H:%M").time()
        e = datetime.strptime(end.strip(),   "%H:%M").time()
    except ValueError:
        return False, "Times must be in HH:MM format (e.g. 14:00)."
    if s >= e:
        return False, f"Start time ({start}) must be before end time ({end})."
    return True, ""


def _default_schedule() -> Dict[str, Any]:
    return {
        "enforce_window": False,
        "timezone": "UTC",
        "update_expiry_hours": 48,
        "windows": [],
    }


def _resolve(path: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, path)