"""Unit tests for the /healthz connection-pool and wire-traffic helpers."""

import asyncio
from types import SimpleNamespace as NS

import pytest

from fluxrt.webrtc.stats import NetStats, connection_pool_stats, summarize_report


def test_empty_pool():
    s = connection_pool_stats([])
    assert s == {"total": 0, "active": 0, "by_state": {}}


def test_counts_and_active():
    s = connection_pool_stats(
        ["connected", "connected", "connecting", "failed"]
    )
    assert s["total"] == 4
    assert s["active"] == 2  # only "connected" counts as active
    assert s["by_state"] == {"connected": 2, "connecting": 1, "failed": 1}


def test_none_state_becomes_unknown():
    s = connection_pool_stats([None, "connected"])
    assert s["total"] == 2
    assert s["active"] == 1
    assert s["by_state"]["unknown"] == 1


def test_by_state_ordered_by_lifecycle_then_alpha():
    s = connection_pool_stats(["closed", "new", "connected", "zzz"])
    # known states in lifecycle order first, unknown extras ("zzz") last
    assert list(s["by_state"].keys()) == ["new", "connected", "closed", "zzz"]


# fps_pipeline is 1 / the last frame's time and swings from poll to poll; the
# kiosk sized morphs with it and showed it as "the" fps. The average must be
# frames actually delivered over real time.
def test_average_fps_is_frames_over_time_between_polls():
    from fluxrt.webrtc.stats import AverageFps

    avg = AverageFps(window=3.0)
    assert avg(100, now=0.0) is None  # nothing to compare with yet
    assert avg(105, now=1.0) is None  # base younger than the window
    assert avg(130, now=6.0) == 5.0  # 30 frames in 6 s
    assert avg(190, now=12.0) == 10.0  # base moved on to the 6 s poll
    assert avg(10, now=18.0) is None  # counter went back: the pipeline restarted


def test_average_fps_survives_fast_polling():
    from fluxrt.webrtc.stats import AverageFps

    avg = AverageFps(window=3.0)
    values = [avg(int(t * 8), now=t) for t in [x * 0.2 for x in range(60)]]
    assert values[-1] is not None and abs(values[-1] - 8.0) < 0.5
    assert len(avg.samples) < 40  # old samples are dropped


# ── wire traffic (NetStats) ───────────────────────────────────────────────────
# Reports mimic aiortc's RTCStatsReport: a dict of id -> stats dataclass.


def _report(rx, tx, rtt=None, lost=None, jitter=None):
    r = {"transport_1": NS(type="transport", id="transport_1", bytesReceived=rx, bytesSent=tx)}
    if rtt is not None or lost is not None:
        r["remote-inbound-rtp_1"] = NS(type="remote-inbound-rtp", kind="video", roundTripTime=rtt, fractionLost=lost)
    if jitter is not None:
        r["inbound-rtp_1"] = NS(type="inbound-rtp", kind="video", jitter=jitter)
        r["inbound-rtp_2"] = NS(type="inbound-rtp", kind="audio", jitter=48_000)  # audio jitter ignored
    return r


def _conn(rx, tx, **q):
    return summarize_report(_report(rx, tx, **q))


def test_summarize_uses_transport_bytes_and_worst_quality():
    s = summarize_report(_report(1_000, 2_000, rtt=0.0234, lost=26, jitter=270))
    assert (s["rx"], s["tx"]) == (1_000, 2_000)
    assert s["rtt_ms"] == pytest.approx(23.4)
    assert s["loss_pct"] == pytest.approx(26 / 256 * 100)
    assert s["jitter_ms"] == pytest.approx(3.0)  # 270 ticks of the 90 kHz video clock
    # no RTCP yet: quality unknown, bytes still counted
    assert summarize_report(_report(5, 6)) == {"rx": 5, "tx": 6, "rtt_ms": None, "loss_pct": None, "jitter_ms": None}


def test_rates_need_a_sample_window_old():
    n = NetStats(window=3.0)
    first = n.update({1: _conn(0, 0)}, now=0.0)
    assert first["in_mbps"] is None and first["window_s"] is None
    assert n.update({1: _conn(500_000, 1_000_000)}, now=1.0)["in_mbps"] is None  # 1 s < window
    s = n.update({1: _conn(4_000_000, 2_000_000)}, now=4.0)
    # base = the newest sample at least 3 s old (t=1): 3.5 MB in, 1 MB out over 3 s
    assert s["in_mbps"] == 9.333
    assert s["out_mbps"] == 2.667
    assert s["window_s"] == 3.0


def test_many_pollers_keep_the_window():
    n = NetStats(window=3.0)
    for t in range(0, 11):  # a poll every second
        s = n.update({1: _conn(t * 1_000_000, 0)}, now=float(t))
    assert s["in_mbps"] == 8.0
    assert 3.0 <= s["window_s"] < 4.0


def test_closed_connection_folds_into_totals_no_negative_rate():
    n = NetStats(window=1.0)
    n.update({1: _conn(1_000, 2_000), 2: _conn(10, 20)}, now=0.0)
    s = n.update({2: _conn(10, 20)}, now=2.0)  # connection 1 closed
    assert (s["bytes_in"], s["bytes_out"]) == (1_010, 2_020)
    assert s["in_mbps"] == 0.0


def test_reused_id_with_fresh_counters_is_a_new_connection():
    n = NetStats(window=1.0)
    n.update({7: _conn(9_000, 9_000)}, now=0.0)
    s = n.update({7: _conn(100, 50)}, now=2.0)  # same id(), counters started over
    assert (s["bytes_in"], s["bytes_out"]) == (9_100, 9_050)
    assert s["in_mbps"] >= 0


def test_worst_quality_across_connections():
    n = NetStats()
    s = n.update({1: _conn(0, 0, rtt=0.010, lost=0), 2: _conn(0, 0, rtt=0.080, lost=64)}, now=0.0)
    assert s["rtt_ms"] == 80.0
    assert s["loss_pct"] == 25.0


class _Pc:
    def __init__(self, report=None, fail=False):
        self.report, self.fail = report, fail

    async def getStats(self):
        if self.fail:
            raise RuntimeError("transport closed")
        return self.report


def test_read_gathers_every_peer_and_keeps_a_failing_one():
    n = NetStats(window=0.0)
    a, b = _Pc(_report(1_000, 1_000)), _Pc(_report(500, 500))
    s = asyncio.run(n.read({a, b}))
    assert (s["bytes_in"], s["bytes_out"]) == (1_500, 1_500)
    b.fail = True  # mid-teardown: its counts stay, not folded and re-added later
    s = asyncio.run(n.read({a, b}))
    assert s["bytes_in"] == 1_500
    b.fail, b.report = False, _report(600, 600)
    assert asyncio.run(n.read({a, b}))["bytes_in"] == 1_600
