"""
Output frame-rate limit of one viewer's track (torch-free, unit-tested).

A track sends every frame the pipeline publishes, interpolated ones included,
unless the viewer asks for a limit. A limit of 0 means "no limit".
"""

MAX_OUT_FPS = 240


def parse_out_fps(raw: str) -> int:
    """Payload of the ctrl message `out-fps:<n>` → frames per second, 0 = no
    limit. Raises ValueError unless it is a whole number in 0..MAX_OUT_FPS."""
    fps = int(raw)
    if fps < 0 or fps > MAX_OUT_FPS:
        raise ValueError("out-fps out of range")
    return fps


def send_wait(last_send: float, now: float, fps: int) -> float:
    """Seconds a track still has to wait before it may send again. With no
    limit (fps 0) it never waits: every published frame goes out."""
    if fps <= 0:
        return 0.0
    return max(0.0, last_send + 1.0 / fps - now)
