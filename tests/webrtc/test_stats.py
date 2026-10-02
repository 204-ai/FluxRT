"""Unit tests for the /healthz connection-pool stats helper."""

from fluxrt.webrtc.stats import connection_pool_stats


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
