"""
Input-ownership state machine + per-peer input consumer for the FluxRT WebRTC
server.

The oldest-connected peer with a live video track steers the pipeline input;
every other peer views the output. When the steering peer leaves, the oldest
waiter takes over; if none remain, the local server camera resumes (or the
output holds its last frame under --no-server-camera).

This module is torch/aiortc/FastAPI-free *to import* (aiortc's MediaStreamError
is imported with a fallback) so the ownership transitions and the recv policy
are unit-testable with fake tracks and fake peer-connection objects.

recv POLICY (this is the bug class that regressed in c855950):
- An OWNER is NEVER evicted because of a frame gap. A healthy owner legitimately
  produces no frame for >5s (first keyframe after claiming, ICE/DTLS/TURN settle,
  a brief stall, a paused camera). Death is detected out-of-band: the caller's
  connectionstatechange handler cancels this task when the pc goes terminal.
- A WAITER is bounded only on its FIRST frame: a peer that never delivers a frame
  AND is not in the 'connected' ICE state is a dead reconnect and is dropped after
  WAITER_FIRST_FRAME_DEADLINE. Once it has delivered a frame, it is never evicted
  on a gap.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import threading
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

try:  # aiortc is present on the server, absent in unit tests.
    from aiortc.mediastreams import MediaStreamError
except Exception:  # pragma: no cover - exercised only when aiortc is installed

    class MediaStreamError(Exception):
        """Fallback so this module imports without aiortc (tests)."""


# ── recv policy ───────────────────────────────────────────────────────────────
# Terminal connection states: the pc is dead, ownership/waiter slot must release.
TERMINAL_STATES = ("failed", "closed", "disconnected")
# How long a waiter that has NEVER delivered a frame may stay before it is treated
# as a dead reconnect (only if it is also not 'connected').
WAITER_FIRST_FRAME_DEADLINE = 25.0


def owner_should_release(connection_state: str) -> bool:
    """An owner is released ONLY on a terminal connection state, never on a frame
    gap. (c855950 evicted on a blind 5s recv timeout regardless of state — that
    froze healthy owners; this is the fix.)"""
    return connection_state in TERMINAL_STATES


def waiter_should_evict(
    connection_state: str, got_first_frame: bool, deadline_passed: bool
) -> bool:
    """A waiter is evicted only if it has never delivered a frame, its first-frame
    deadline passed, and it is not currently 'connected'. A connected-but-slow
    waiter keeps waiting; a waiter that delivered a frame is never gap-evicted."""
    return (not got_first_frame) and deadline_passed and connection_state != "connected"


def _conn_state(pc) -> str:
    return getattr(pc, "connectionState", "connected")


@dataclass(frozen=True)
class ReleaseOutcome:
    """Result of release(): the I/O (broadcasts) is the caller's job, the decision
    lives here."""

    had_owner: bool
    became_idle: bool  # ownership went None AND no waiters remain
    server_camera_resumes: bool


class InputOwnership:
    """Single owner of the input-steering state. All of input_owner / waiters /
    the active flag / the seq counter are mutated only here, under one lock — so
    a phantom consumer can never pin the active flag against a dead pc."""

    def __init__(self, has_server_camera: bool = True):
        self._lock = threading.Lock()
        self._owner = None  # pc identity, or None
        self._waiters: dict[int, object] = {}  # seq -> pc
        self._seq = itertools.count()
        self._active = threading.Event()  # the producer thread waits on this
        self.has_server_camera = has_server_camera

    def register_waiter(self, pc) -> int:
        with self._lock:
            seq = next(self._seq)
            self._waiters[seq] = pc
            return seq

    def try_claim(self, seq: int, pc) -> bool:
        """Become owner iff no one owns and this seq is the oldest waiter.
        Idempotent if already owner."""
        with self._lock:
            if self._owner is pc:
                return True
            if self._owner is None and self._waiters and seq == min(self._waiters):
                self._owner = pc
                self._active.set()
                return True
            return False

    def release(self, seq: int, pc) -> ReleaseOutcome:
        """The ONLY path that clears ownership / the active flag."""
        with self._lock:
            self._waiters.pop(seq, None)
            had_owner = self._owner is pc
            if had_owner:
                self._owner = None
            became_idle = False
            server_camera_resumes = False
            if self._owner is None and not self._waiters and self._active.is_set():
                self._active.clear()
                became_idle = True
                server_camera_resumes = self.has_server_camera
            return ReleaseOutcome(had_owner, became_idle, server_camera_resumes)

    def owner_is(self, pc) -> bool:
        with self._lock:
            return self._owner is pc

    def is_active(self) -> bool:
        return self._active.is_set()

    def active_event(self) -> threading.Event:
        """The same Event the local camera producer thread waits on."""
        return self._active

    def num_waiters(self) -> int:
        with self._lock:
            return len(self._waiters)


# Type of the frame sink: an async callable that consumes one decoded VideoFrame
# (the caller offloads decode + pipeline drive to an executor inside it).
FrameSink = Callable[[object], Awaitable[None]]
# Role-broadcast hook: notify(event, pc, outcome=None); event in {"claimed","released"}.
NotifyHook = Callable[..., None]


async def _pump_owner_frames(track, pc, sink: FrameSink, log=None) -> None:
    """Drain-to-latest: a reader task always overwrites `latest`, the processing
    loop drives only the newest frame. Bounds round-trip lag and memory when the
    pipeline is slower than the peer camera. The owner reader BLOCKS on recv()
    with no timeout — a frame gap never evicts a live owner (the caller cancels
    this task when the pc goes terminal)."""
    latest = [None]
    new_frame = asyncio.Event()
    stopped = asyncio.Event()

    async def _reader():
        try:
            while True:
                latest[0] = await track.recv()
                new_frame.set()
        except MediaStreamError:
            pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            if log:
                log.warning("Owner track recv error: %s", exc)
        finally:
            stopped.set()
            new_frame.set()  # wake the processing loop so it can exit

    reader = asyncio.ensure_future(_reader())
    try:
        while not (stopped.is_set() and latest[0] is None):
            await new_frame.wait()
            new_frame.clear()
            frame, latest[0] = latest[0], None
            if frame is None:
                continue
            await sink(frame)
    finally:
        reader.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reader


async def consume_peer_input(
    track,
    pc,
    ownership: InputOwnership,
    sink: FrameSink,
    *,
    notify: Optional[NotifyHook] = None,
    log=None,
    first_frame_deadline: float = WAITER_FIRST_FRAME_DEADLINE,
) -> None:
    """Pull VideoFrames from a remote track and (when this peer owns the input)
    feed them into the pipeline via `sink`. Waits for ownership while draining
    its track view-only; the oldest waiter takes over on release."""
    notify = notify or (lambda *a, **k: None)
    loop = asyncio.get_running_loop()
    seq = ownership.register_waiter(pc)

    try:
        # ── wait for ownership, draining frames so the inbound queue can't grow ──
        got_first_frame = False
        deadline = loop.time() + first_frame_deadline
        announced_waiting = False
        while not ownership.try_claim(seq, pc):
            if not announced_waiting:
                announced_waiting = True
                if log:
                    log.info("Peer %x (seq %d) waiting — view-only", id(pc), seq)
            if got_first_frame:
                # Delivered a frame already: a real, connected view-only peer.
                # Block (never gap-evict); death is handled by the caller's
                # connectionstatechange -> task cancel.
                try:
                    await track.recv()
                except MediaStreamError:
                    return
                continue
            remaining = deadline - loop.time()
            if waiter_should_evict(_conn_state(pc), got_first_frame, remaining <= 0):
                if log:
                    log.info("Peer %x (seq %d) never connected — dropping", id(pc), seq)
                return
            try:
                await asyncio.wait_for(track.recv(), timeout=max(1.0, remaining))
                got_first_frame = True
            except asyncio.TimeoutError:
                continue  # re-check claim + liveness; do NOT evict a connected peer
            except MediaStreamError:
                return

        # ── now the owner ──
        if log:
            log.info("Peer %x (seq %d) now drives input", id(pc), seq)
        notify("claimed", pc)
        await _pump_owner_frames(track, pc, sink, log=log)
    finally:
        outcome = ownership.release(seq, pc)
        if log and outcome.had_owner:
            log.info("Peer %x (seq %d) released input", id(pc), seq)
        notify("released", pc, outcome)
