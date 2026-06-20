"""Tests for reap_process — the bounded subprocess kill ladder + the
multiprocessing._children detach that prevents the no-timeout atexit-join hang.
"""

import multiprocessing as mp

import _child_targets
import psutil

from fluxrt.webrtc.proc import reap_process


def _spawn(ignore_sigterm):
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_child_targets.busy_child, args=(ignore_sigterm,))
    p.start()
    return p


def test_reap_killable_child():
    p = _spawn(ignore_sigterm=False)
    assert psutil.pid_exists(p.pid)
    ok = reap_process(p, join_timeouts=(0.2, 0.5, 1.0))
    assert ok is True
    assert not p.is_alive()


def test_reap_sigterm_ignoring_child_escalates_to_sigkill():
    p = _spawn(ignore_sigterm=True)  # ignores SIGTERM -> terminate() is a no-op
    assert psutil.pid_exists(p.pid)
    ok = reap_process(p, join_timeouts=(0.2, 0.3, 1.5))
    assert ok is True, "SIGKILL escalation should reap a SIGTERM-ignoring child"
    assert not p.is_alive()


class _ImmortalProc:
    """A process that never dies — to exercise the survivor-detach path."""

    def __init__(self):
        self.pid = 2_147_483_001  # almost certainly nonexistent; os.kill -> suppressed
        self.terminated = 0
        self.killed = 0

    def is_alive(self):
        return True

    def join(self, timeout=None):
        return None

    def terminate(self):
        self.terminated += 1

    def kill(self):
        self.killed += 1


def test_survivor_is_detached_from_multiprocessing_children():
    import multiprocessing.process as mpp

    fake = _ImmortalProc()
    mpp._children.add(fake)  # simulate a registered, unkillable child
    try:
        ok = reap_process(fake, join_timeouts=(0, 0, 0))
        assert ok is False
        assert fake.terminated == 1 and fake.killed == 1
        # The whole point: removed so atexit's no-timeout join can't block on it.
        assert fake not in mpp._children
    finally:
        mpp._children.discard(fake)
