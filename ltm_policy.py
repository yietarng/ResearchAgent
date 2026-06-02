from __future__ import annotations

from enum import Enum
from typing import Any


class StorageDecision(Enum):
    STORE = "store"
    SKIP_NO_SIGNAL = "skip_no_signal"
    SKIP_TOO_SHORT = "skip_too_short"


# Keywords that signal storable, reusable content for agent-initiated writes.
# Kept narrow — common words like "summary", "report", "project", "always"
# were removed because they appear in almost every result and make the
# policy a near-passthrough.  Node-level writes (write_ltm) bypass this
# check via force_store=True and are always stored if length passes.
_STORABLE_SIGNALS = [
    "survey",
    "benchmark",
    "interest",
    "prefer",
    "reference",
    "finding",
    "conclusion",
]

# Minimum character length for stored content
_MIN_CHARS = 80


def should_store_in_ltm(
    content: str,
    source: str = "unknown",
    metadata: dict[str, Any] | None = None,
) -> tuple[bool, StorageDecision]:
    """Return (should_store, decision_reason).

    Duplicate detection is handled by LTMStore.store() via a content_hash
    metadata field — no in-process cache needed here.
    The source parameter is accepted for call-site compatibility but is
    not used for filtering (no source-based block list is maintained).
    """
    metadata = metadata or {}

    if len(content.strip()) < _MIN_CHARS:
        return False, StorageDecision.SKIP_TOO_SHORT

    lower = content.lower()
    has_signal = any(kw in lower for kw in _STORABLE_SIGNALS)
    is_explicit = metadata.get("force_store", False)

    if not has_signal and not is_explicit:
        return False, StorageDecision.SKIP_NO_SIGNAL

    return True, StorageDecision.STORE
