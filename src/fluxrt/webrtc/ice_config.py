"""Which STUN server the WebRTC peers use (FLUXRT_STUN). Torch- and aiortc-free
so it can be unit-tested on a laptop.

A show machine runs without internet: its kiosk is on the same machine and needs
no STUN, and a STUN lookup with an IP but no internet delays every answer by up
to 5 s (aioice waits that long for the server-reflexive candidate). "none",
"off" or an empty value mean no STUN server. An empty value is not enough on
Windows, where setting an env var to "" deletes it, so start_show.ps1 uses "none".
"""

DEFAULT_STUN = "stun:stun.l.google.com:19302"


def stun_urls(value: str | None) -> list[str]:
    """STUN URLs for the FLUXRT_STUN value; None (unset) = the public default."""
    if value is None:
        return [DEFAULT_STUN]
    value = value.strip()
    if value.lower() in ("", "none", "off"):
        return []
    return [value]
