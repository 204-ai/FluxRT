"""Connection-pool and wire-traffic stats for /healthz.

Torch-free and pure so it is unit-tested off-GPU: given the connectionState of
every peer in the pool, summarise total / active / per-state counts; given
each peer's getStats() report, total the bytes on the wire and rate them.
"""

from __future__ import annotations

import asyncio
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


def summarize_report(report) -> dict:
    """One connection's `RTCPeerConnection.getStats()` → wire bytes + quality.

    Bytes come from the DTLS transport's counters, not the RTP streams: they
    are what crosses the network (SRTP, RTCP, DTLS, data channel), and aiortc's
    inbound-rtp stats carry no byte count at all. Quality is the worst stream:
      rtt_ms     smoothed round trip from RTCP receiver reports (s → ms)
      loss_pct   share of what we send that the peer reports lost (RTCP's
                 8-bit fraction, /256)
      jitter_ms  arrival jitter of incoming video (RTP 90 kHz clock → ms)
    Each is None until the first RTCP report arrives.
    """
    rx = tx = 0
    rtt = loss = jitter = None
    for s in report.values():
        kind = getattr(s, "type", None)
        if kind == "transport":
            rx += int(getattr(s, "bytesReceived", 0) or 0)
            tx += int(getattr(s, "bytesSent", 0) or 0)
        elif kind == "remote-inbound-rtp":
            if getattr(s, "roundTripTime", None) is not None:
                rtt = max(rtt or 0.0, s.roundTripTime * 1000.0)
            if getattr(s, "fractionLost", None) is not None:
                loss = max(loss or 0.0, s.fractionLost / 256.0 * 100.0)
        elif kind == "inbound-rtp" and getattr(s, "kind", None) == "video":
            if getattr(s, "jitter", None) is not None:
                jitter = max(jitter or 0.0, s.jitter / 90.0)
    return {"rx": rx, "tx": tx, "rtt_ms": rtt, "loss_pct": loss, "jitter_ms": jitter}


class NetStats:
    """Wire traffic across every peer connection, for /healthz.

    A connection's transport counters vanish with it, so a closed connection's
    last counts fold into a base and `bytes_in` / `bytes_out` only ever grow.
    Mbps comes from the newest sample at least `window` seconds back (the
    AverageFps idea): several pollers don't shrink the window, and the rates
    are None until such a sample exists.
    """

    def __init__(self, window: float = 3.0):
        self.window = window
        self._last: dict[int, tuple[int, int]] = {}
        self._base = (0, 0)
        self.samples: list[tuple[float, int, int]] = []

    async def read(self, pcs) -> dict:
        conns = list(pcs)
        reports = await asyncio.gather(*(pc.getStats() for pc in conns), return_exceptions=True)
        per: dict[int, dict] = {}
        for pc, report in zip(conns, reports):
            key = id(pc)
            if not isinstance(report, BaseException):
                per[key] = summarize_report(report)
            elif key in self._last:
                # a connection mid-teardown can fail getStats: keep its counts, don't fold it yet
                rx, tx = self._last[key]
                per[key] = {"rx": rx, "tx": tx, "rtt_ms": None, "loss_pct": None, "jitter_ms": None}
        return self.update(per)

    def update(self, per: dict[int, dict], now: float | None = None) -> dict:
        """Fold, total and rate one poll's per-connection summaries (pure; tested)."""
        now = time.monotonic() if now is None else now
        base_rx, base_tx = self._base
        for key, (rx, tx) in self._last.items():
            cur = per.get(key)
            # gone, or the id now names a new connection whose counters started over
            if cur is None or cur["rx"] < rx or cur["tx"] < tx:
                base_rx += rx
                base_tx += tx
        self._base = (base_rx, base_tx)
        self._last = {k: (c["rx"], c["tx"]) for k, c in per.items()}
        total_rx = base_rx + sum(c["rx"] for c in per.values())
        total_tx = base_tx + sum(c["tx"] for c in per.values())

        self.samples.append((now, total_rx, total_tx))
        while len(self.samples) > 2 and now - self.samples[1][0] >= self.window:
            self.samples.pop(0)
        then, rx0, tx0 = self.samples[0]
        dt = now - then
        ready = dt > 0 and dt >= self.window

        def mbps(cur: int, old: int):
            return round((cur - old) * 8 / dt / 1e6, 3) if ready else None

        def worst(field: str, digits: int):
            vals = [c[field] for c in per.values() if c.get(field) is not None]
            return round(max(vals), digits) if vals else None

        return {
            "in_mbps": mbps(total_rx, rx0),
            "out_mbps": mbps(total_tx, tx0),
            "bytes_in": total_rx,
            "bytes_out": total_tx,
            "rtt_ms": worst("rtt_ms", 1),
            "loss_pct": worst("loss_pct", 2),
            "jitter_ms": worst("jitter_ms", 1),
            "window_s": round(dt, 1) if ready else None,
        }
