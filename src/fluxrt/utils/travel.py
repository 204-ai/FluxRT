"""Prompt-travel (morph) length: generated frames, or seconds on the clock.

A morph is slower per frame than normal running, so one counted in frames
takes longer the slower the machine is: a "4 s" morph sized for 8 fps ran 7 s
at 4.5 fps, most of the time between two prompts. Timed on the clock it ends
when it should, however many frames that turns out to be.
"""

BATCH_FPS = 8  # frames a duration stands for where no clock applies (batch render)


def parse_length(raw):
    """(frames, seconds) for a length field: "48" or 48 = generated frames,
    "4s" / "2.5s" = seconds (frames is then the count a batch render uses).
    None when it is neither."""
    text = str(raw).strip()
    if text.isdigit():
        return (int(text), None) if int(text) >= 1 else None
    if text.endswith("s"):
        try:
            seconds = float(text[:-1])
        except ValueError:
            return None
        if 0 < seconds <= 120:
            return max(1, round(seconds * BATCH_FPS)), seconds
    return None


def progress(step: int, frames: int, seconds, elapsed: float, frame_time: float) -> float:
    """Blend position (0..1] of the frame generated now. `step` counts from 1.
    Timed morphs aim one frame time ahead: that is when this frame is shown."""
    if seconds:
        return min(1.0, (elapsed + frame_time) / seconds)
    return min(1.0, step / frames)
