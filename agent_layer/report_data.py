"""Report data access — how the Agent layer loads a report window.

Reports must select episodes **by time range**, never by "the newest N rows".
A row cap silently truncates a busy week: the record total, the tier
distribution, and the representative event all under-count, while the sensing
and event tables are queried by full time range — so the report contradicts
itself.

This module owns that selection so the correct window is guaranteed for every
caller, including callers that never pass a pre-filtered episode list. It is
the only place in the Agent layer that reaches into storage for report data;
`report_agent` itself stays pure and provider-free.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from storage.models import (
    query_episodes_by_resident,
    query_episodes_in_range,
    query_latest_sensing_window,
)

DEFAULT_WINDOW_DAYS = 7


@dataclass
class ReportWindow:
    """A complete, time-bounded slice of triage history."""

    resident_id: str
    reference_ts: datetime
    start: str
    end: str
    window_days: int
    episodes: list = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.episodes)


def resolve_reference_ts(
    resident_id: str,
    explicit: Optional[datetime] = None,
) -> datetime:
    """Pick the anchor time a report is written "as of".

    Replay and seeded demo data can be old, so anchoring on wall-clock time
    would produce an empty report. Anchor on the newest available evidence
    instead: the latest sensing window, falling back to the newest episode.
    """
    if explicit is not None:
        return explicit.replace(tzinfo=None) if explicit.tzinfo else explicit

    latest_window = query_latest_sensing_window(resident_id) or {}
    newest_episode = query_episodes_by_resident(resident_id, 1)

    candidates: list[datetime] = []
    raw_values = [
        latest_window.get("timestamp"),
        newest_episode[0].get("start_time") if newest_episode else None,
    ]
    for raw in raw_values:
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(str(raw))
        except (ValueError, TypeError):
            continue
        candidates.append(dt.replace(tzinfo=None) if dt.tzinfo else dt)

    return max(candidates) if candidates else datetime.now()


def load_report_window(
    resident_id: str,
    reference_ts: Optional[datetime] = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> ReportWindow:
    """Load every episode inside the report window.

    The episode query is deliberately uncapped: correctness comes from the
    time bounds, not from a row limit. The anchor lookup above uses a limit of
    1 because it only needs the most recent row, not a page of them.

    Args:
        resident_id: resident key.
        reference_ts: anchor time; resolved from the newest evidence if None.
        window_days: lookback length in days.

    Returns:
        A :class:`ReportWindow` whose ``episodes`` are ordered oldest-first.
    """
    ref = resolve_reference_ts(resident_id, reference_ts)
    start = (ref - timedelta(days=window_days)).isoformat()
    end = ref.isoformat()

    episodes = query_episodes_in_range(resident_id, start, end)
    for ep in episodes:
        _parse_json_columns(ep)

    return ReportWindow(
        resident_id=resident_id,
        reference_ts=ref,
        start=start,
        end=end,
        window_days=window_days,
        episodes=episodes,
    )


_JSON_COLUMNS = ("decision", "evidence", "action", "audit")


def _parse_json_columns(episode: dict) -> None:
    """Decode the TEXT(JSON) columns in place.

    Storage returns these as strings; the report context accepts either, but
    decoding here means callers can rely on dicts.
    """
    import json

    for key in _JSON_COLUMNS:
        value = episode.get(key)
        if isinstance(value, str):
            try:
                episode[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
