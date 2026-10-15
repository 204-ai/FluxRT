"""FLUXRT_STUN parsing. A show machine has no internet: with a public STUN server
configured, every /offer answer waits up to 5 s for a STUN reply that never comes
(IP without internet), so the show must be able to run with no STUN at all."""

from fluxrt.webrtc.ice_config import DEFAULT_STUN, stun_urls


def test_unset_keeps_public_stun_for_remote_clients():
    assert stun_urls(None) == [DEFAULT_STUN]


def test_none_off_and_empty_mean_no_stun():
    # "none" is what start_show.ps1 sets: on Windows an empty env var is deleted,
    # which would silently bring the public default back.
    for value in ("none", "NONE", "off", "", "  "):
        assert stun_urls(value) == [], value


def test_explicit_server_is_used_as_given():
    assert stun_urls(" stun:lan-stun:3478 ") == ["stun:lan-stun:3478"]
