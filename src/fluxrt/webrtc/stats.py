"""Connection-pool stats for /healthz.

Torch-free and pure so it is unit-tested off-GPU: given the connectionState of
every peer in the pool, summarise total / active / per-state counts.
"""

from __future__ import annotations

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
