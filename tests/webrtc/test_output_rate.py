"""An interpolating pipeline publishes about twice as many frames as it
generates (56 a second on the 4090 box). The output track was pinned to 30, so
the viewer lost about half of them. These assertions hold the fix: a viewer
without a limit gets every published frame, and a limit is a ceiling the viewer
chose, not a constant in the server.
"""

import pytest

from fluxrt.webrtc.output_rate import MAX_OUT_FPS, parse_out_fps, send_wait


def test_no_limit_never_holds_a_frame_back():
    # frames 1 ms apart: with no limit none of them waits, so none is replaced
    # by a newer one before it is sent
    assert send_wait(last_send=10.000, now=10.001, fps=0) == 0.0


def test_limit_spaces_sends_by_its_period():
    # 56 published a second against a limit of 30: the next frame arrives
    # 17.9 ms after the last send and has to wait out the rest of 33.3 ms
    wait = send_wait(last_send=10.0, now=10.0 + 1 / 56, fps=30)
    assert wait == pytest.approx(1 / 30 - 1 / 56)


def test_limit_does_not_delay_a_pipeline_slower_than_it():
    # 10 published a second against a limit of 30: output rate == pipeline rate
    assert send_wait(last_send=10.0, now=10.1, fps=30) == 0.0


@pytest.mark.parametrize("raw,fps", [("0", 0), ("30", 30), ("60", 60), (str(MAX_OUT_FPS), MAX_OUT_FPS)])
def test_parse_accepts_no_limit_and_whole_rates(raw, fps):
    assert parse_out_fps(raw) == fps


@pytest.mark.parametrize("raw", ["-1", str(MAX_OUT_FPS + 1), "29.97", "fast", ""])
def test_parse_rejects_what_is_not_a_rate(raw):
    with pytest.raises(ValueError):
        parse_out_fps(raw)
