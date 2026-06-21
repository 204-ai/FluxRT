"""
Bounded subprocess teardown for the spawned inference / output-scheduler
children.

Why this exists: a CUDA-wedged child can survive even SIGKILL+join for a while
(uninterruptible driver syscall, D state). multiprocessing's atexit
_exit_function then joins every still-registered non-daemon child with NO
timeout — a permanent shutdown hang requiring a manual kill. After the kill
escalation we force one more SIGKILL and DISCARD the survivor from
multiprocessing.process._children so that atexit join can never block on it.

Torch-free and import-light so the escalation ladder is unit-testable with a
real, deliberately-wedged multiprocessing child (no GPU).
"""

from __future__ import annotations

import os
import signal


def reap_process(proc, join_timeouts: tuple[float, float, float] = (5.0, 3.0, 2.0)) -> bool:
    """Escalate join -> terminate -> kill on `proc`, bounded by `join_timeouts`.
    The caller must already have signalled the child to stop (e.g. flipped its
    `running` Value) so the first join can succeed gracefully.

    Returns True if the process is confirmed dead, False if it outlived even the
    final SIGKILL (in which case it has been detached from multiprocessing's
    child registry so interpreter-exit can't block joining it).
    """
    if proc is None:
        return True

    t_join, t_term, t_kill = join_timeouts

    proc.join(timeout=t_join)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=t_term)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=t_kill)

    if proc.is_alive():
        # Survived SIGKILL+join (typically a D-state CUDA syscall). Re-signal and
        # DETACH from multiprocessing._children so the no-timeout atexit join
        # can't hang the parent and leave an orphaned GPU process.
        pid = getattr(proc, "pid", None)
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        _discard_from_children(proc)
        return False
    return True


def _discard_from_children(proc) -> None:
    try:
        import multiprocessing.process as _mpp

        _mpp._children.discard(proc)
    except Exception:
        pass
