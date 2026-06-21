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
