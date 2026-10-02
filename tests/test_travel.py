"""Prompt-travel length: frames or seconds (src/fluxrt/utils/travel.py)."""

import pytest

from fluxrt.utils import travel


def test_frames_and_seconds_are_told_apart():
    assert travel.parse_length("48") == (48, None)
    assert travel.parse_length(48) == (48, None)
    assert travel.parse_length("4s") == (32, 4.0)  # frames: what a batch render uses
    assert travel.parse_length("2.5s") == (20, 2.5)


@pytest.mark.parametrize("raw", ["0", "-3", "abc", "s", "0s", "-1s", "500s", "", "4.5", None])
def test_anything_else_is_rejected(raw):
    assert travel.parse_length(raw) is None


# The reason seconds exist: a morph must end on the clock however slow the
# frames are. Counted in frames, a 32-frame morph at 4 fps takes 8 s.
def test_a_timed_morph_ends_on_the_clock_whatever_the_frame_rate():
    for frame_time in (0.05, 0.25):  # 20 fps and 4 fps
        elapsed, step, shown = 0.0, 0, []
        while True:
            step += 1
            t = travel.progress(step, 32, 4.0, elapsed, frame_time)
            shown.append(t)
            if t >= 1.0:
                break
            elapsed += frame_time
        assert abs(elapsed + frame_time - 4.0) <= frame_time  # lands at 4 s
        assert shown[0] > 0  # the first frame already shows progress
        assert shown == sorted(shown)


def test_a_counted_morph_takes_exactly_its_frames():
    ts = [travel.progress(i, 4, None, 123.0, 0.1) for i in range(1, 6)]
    assert ts == [0.25, 0.5, 0.75, 1.0, 1.0]
