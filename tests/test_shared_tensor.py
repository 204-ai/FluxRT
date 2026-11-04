"""SharedTensor's seqlock: layout compatibility, the write counter, and a
cross-process stress test that a reader never gets half of one frame and
half of the next (the unversioned tensor does, see `torn_frames`)."""

import multiprocessing as mp
import time

import numpy as np
import pytest

import _shm_writer
from fluxrt.utils.shared_tensor import SharedTensor

SHAPE = (640, 1152, 3)  # the flow-upscaler output frame, 2.2 MB


def test_payload_stays_at_offset_zero_for_old_style_readers():
    t = SharedTensor((4, 5, 3), create=True)
    try:
        t.copy_from(np.arange(60, dtype=np.uint8).reshape(4, 5, 3))
        # what code that predates the trailer does: map the payload by name
        from multiprocessing import shared_memory

        shm = shared_memory.SharedMemory(name=t.name)
        old = np.ndarray((4, 5, 3), dtype=np.uint8, buffer=shm.buf)
        assert np.array_equal(old, t.to_numpy())
        del old
        shm.close()
    finally:
        t.close_and_unlink()


def test_attaching_by_name_finds_the_trailer():
    t = SharedTensor((8, 8, 3), create=True)
    try:
        other = SharedTensor((8, 8, 3), name=t.name)
        assert t.versioned and other.versioned
        other.copy_from(np.full((8, 8, 3), 7, np.uint8))
        assert t.sequence() == 2 and int(t.to_numpy()[0, 0, 0]) == 7
        other.close()
    finally:
        t.close_and_unlink()


def test_unversioned_tensor_has_no_trailer():
    t = SharedTensor((8, 8, 3), create=True, versioned=False)
    try:
        assert not t.versioned and t.sequence() == 0
        t.copy_from(np.ones((8, 8, 3), np.uint8))
        assert t.to_numpy().sum() == 8 * 8 * 3
    finally:
        t.close_and_unlink()


def test_the_counter_is_odd_while_writing_and_advances_by_two():
    t = SharedTensor((2, 2, 3), create=True)
    try:
        with t.writing() as a:
            assert t.sequence() % 2 == 1
            a[:] = 3
        assert t.sequence() == 2
        t.copy_from(np.zeros((2, 2, 3), np.uint8))
        assert t.sequence() == 4
    finally:
        t.close_and_unlink()


def test_a_writer_that_died_mid_write_does_not_wedge_readers():
    t = SharedTensor((2, 2, 3), create=True)
    try:
        t._seq[0] = 5  # odd forever: the writer is gone
        start = time.perf_counter()
        assert t.read_into(np.empty((2, 2, 3), np.uint8)) is False
        assert time.perf_counter() - start < 0.5
        t.copy_from(np.zeros((2, 2, 3), np.uint8))  # the next write recovers
        assert t.sequence() % 2 == 0
        assert t.read_into(np.empty((2, 2, 3), np.uint8)) is True
    finally:
        t.close_and_unlink()


def torn_frames(versioned: bool, gap: float, seconds: float = 1.5) -> dict:
    """A reader racing a writer in another process that writes a frame every
    `gap` seconds (0: back to back), counted over the same window: reads,
    writes, torn reads, and torn reads read_into did not report."""
    ctx = mp.get_context("spawn")
    t = SharedTensor(SHAPE, create=True)
    t.copy_from(np.full(SHAPE, 0x55, np.uint8))
    stop, writes = ctx.Event(), ctx.Value("Q", 0)
    p = ctx.Process(target=_shm_writer.write_alternating, args=(t.name, SHAPE, versioned, stop, writes, gap))
    p.start()
    reader = SharedTensor(SHAPE, name=t.name)
    if not versioned:
        reader._seq = None
    out = np.empty(SHAPE, np.uint8)
    n = {"reads": 0, "torn": 0, "silently_torn": 0}
    try:
        deadline = time.perf_counter() + 30
        while writes.value == 0:  # the writer is up and writing
            assert p.is_alive() and time.perf_counter() < deadline, "writer did not start"
            time.sleep(0.01)
        written0 = writes.value
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            clean = reader.read_into(out)
            n["reads"] += 1
            if out.min() != out.max():
                n["torn"] += 1
                n["silently_torn"] += clean
        n["writes"] = writes.value - written0
    finally:
        stop.set()
        p.join(10)
        reader.close()
        t.close_and_unlink()
    return n


# A write every millisecond: a duty cycle ~10x what FluxRT's writers have
# (one 0.05-2 MB frame per frame time).
BUSY_WRITER_GAP = 0.001


def test_no_torn_frames_across_processes():
    n = torn_frames(versioned=True, gap=BUSY_WRITER_GAP)
    assert n["reads"] > 50 and n["writes"] > 50, n
    assert n["torn"] == 0, n


@pytest.mark.skipif(mp.cpu_count() < 2, reason="the race needs two cores")
def test_the_unversioned_path_does_tear():
    """The failure the seqlock removes, so the stress test above is known to
    be able to see it (measured: ~6% of reads at this gap, 0.6% at 60 Hz)."""
    n = torn_frames(versioned=False, gap=BUSY_WRITER_GAP)
    assert n["writes"] > 50, n
    assert n["torn"] > 0, n


@pytest.mark.skipif(mp.cpu_count() < 2, reason="the race needs two cores")
def test_a_reader_starved_by_a_back_to_back_writer_says_so():
    """Without a pause between writes a reader cannot get a clean copy; it
    gives up after its time budget and reports it rather than returning a
    torn frame as if it were whole."""
    n = torn_frames(versioned=True, gap=0.0, seconds=0.5)
    assert n["silently_torn"] == 0, n
