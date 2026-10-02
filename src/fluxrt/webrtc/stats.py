"""Connection-pool stats for /healthz.

Torch-free and pure so it is unit-tested off-GPU: given the connectionState of
every peer in the pool, summarise total / active / per-state counts.
"""

from __future__ import annotations

import time
from collections import Counter

# aiortc RTCPeerConnection.connectionState values.
_KNOWN_STATES = ("new", "connecting", "connected", "disconnected", "failed", "closed")


def connection_pool_stats(states) -> dict:
    """Summarise the peer-connection pool.

    Args:
        states: iterable of per-peer connectionState strings (None -> "unknown").

    Returns:
        {"total": int, "active": int, "by_state": {state: count, ...}} where
        `active` counts peers in the "connected" state. by_state is sorted with
        known states first (in lifecycle order) then any extras alphabetically.
    """
    counts = Counter((s or "unknown") for s in states)
    order = {s: i for i, s in enumerate(_KNOWN_STATES)}
    by_state = {
        s: counts[s]
        for s in sorted(counts, key=lambda s: (order.get(s, len(order)), s))
    }
    return {
        "total": sum(counts.values()),
        "active": counts.get("connected", 0),
        "by_state": by_state,
    }


class AverageFps:
    """Generated frames per second since an earlier poll at least `window`
    seconds back: what the pipeline really delivered. One frame's 1 / time
    (fps_pipeline) swings between 3 and 30 from poll to poll."""

    def __init__(self, window: float = 3.0):
        self.window = window
        self.samples: list[tuple[float, int]] = []

    def __call__(self, frames: int, now: float | None = None):
        """Record the frame counter; the average fps, or None until a sample
        `window` seconds old exists (or the counter went back: a restart)."""
        now = time.monotonic() if now is None else now
        self.samples.append((now, frames))
        # keep the newest sample that is at least `window` old as the base
        while len(self.samples) > 2 and now - self.samples[1][0] >= self.window:
            self.samples.pop(0)
        then, frames_then = self.samples[0]
        if now - then < self.window or frames < frames_then:
            return None
        return round((frames - frames_then) / (now - then), 2)
