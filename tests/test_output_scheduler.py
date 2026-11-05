"""The output scheduler's pacing, run in-process on real shared memory with
the model's side played by the test (write the batch, set pack_is_ready,
like ModelInferenceSubprocess.sync_fps_and_send)."""

import threading
import time
from multiprocessing import Value

import numpy as np

from fluxrt.stream_processor.output_scheduler_subprocess import OutputSchedulerSubprocess
from fluxrt.utils.shared_tensor import SharedTensor

H, W = 8, 12


class Rig:
    def __init__(self, exp=1):
        config = {"resolution": {"height": H, "width": W}, "interpolation_exp": exp}
        self.batch = SharedTensor((2**exp, H, W, 3), create=True)
        self.out = SharedTensor((H, W, 3), create=True)
        self.ready, self.proc = Value("b", False), Value("f", 0.0)
        self.sched = OutputSchedulerSubprocess(
            config, self.batch.name, self.out.name, self.ready, self.proc, Value("b", False), Value("L", 0)
        )
        self.shown = []  # (value, time) per published frame
        real_publish = self.sched.publish

        def publish(frame):
            real_publish(frame)
            self.shown.append((int(frame[0, 0, 0]), time.perf_counter()))

        self.sched.publish = publish
        self.sched.running.value = True
        self.thread = threading.Thread(target=self.sched.process_main, daemon=True)
        self.thread.start()
        time.sleep(0.02)

    def write_pack(self, values, proc_time):
        self.proc.value = proc_time
        self.batch.copy_from(np.stack([np.full((H, W, 3), v, np.uint8) for v in values]))
        self.ready.value = True
        return time.perf_counter()

    def values(self):
        return [v for v, _ in self.shown]

    def time_of(self, value):
        return next(t for v, t in self.shown if v == value)

    def close(self):
        self.sched.running.value = False
        self.thread.join(2)
        self.batch.close_and_unlink()
        self.out.close_and_unlink()


def test_a_pack_is_paced_over_one_frame_time():
    rig = Rig(exp=2)
    try:
        rig.write_pack([1, 2, 3, 4], proc_time=0.08)
        time.sleep(0.15)
        assert rig.values() == [1, 2, 3, 4]
        span = rig.time_of(4) - rig.time_of(1)
        assert 0.055 < span < 0.09  # three holds of 20 ms
        assert int(rig.out.to_numpy()[0, 0, 0]) == 4
    finally:
        rig.close()


def test_steady_packs_show_every_frame_in_order():
    rig = Rig(exp=1)
    try:
        for k in range(1, 5):
            time.sleep(0.04)
            rig.write_pack([10 * k, 10 * k + 1], proc_time=0.04)
        time.sleep(0.08)
        assert rig.values() == [10, 11, 20, 21, 30, 31, 40, 41]
    finally:
        rig.close()


def test_a_fast_frame_after_a_slow_one_does_not_hide_a_generated_frame():
    """The old loop read batch[i] straight from shared memory while pacing
    and cleared pack_is_ready afterwards: when a frame came in less than half
    the previous frame's time (the spatial cache makes frame times vary 2x),
    the newer pack's generated frame replaced the older one before it was
    shown and the newer pack was then dropped: half the generated frames
    never reached the screen. Now every generated frame is shown and held,
    and the newer pack's interval takes over at once."""
    rig = Rig(exp=1)
    try:
        t0 = rig.write_pack([10, 11], proc_time=0.2)  # in-between, generated; hold 100 ms
        time.sleep(0.02)
        t1 = rig.write_pack([20, 21], proc_time=0.04)  # arrives early; hold 20 ms
        time.sleep(0.12)
        assert rig.values() == [10, 11, 20, 21]
        # 11 is not held for the stale 100 ms: it goes out when the new pack lands
        assert rig.time_of(11) - t1 < 0.01
        # ... and stays up one new interval before the new pack's frames follow
        assert rig.time_of(20) - rig.time_of(11) >= 0.018
        assert rig.time_of(21) - t1 < 0.06
    finally:
        rig.close()


def test_a_pack_s_first_frame_goes_out_on_arrival():
    """Steady packs with a few ms of jitter: the in-between is published as
    the pack lands and the generated frame one interval later, as the old
    loop did (a hold before the first frame delayed them by the jitter)."""
    rig = Rig(exp=1)
    try:
        rng = np.random.default_rng(0)
        arrivals = {}
        for k in range(1, 9):
            time.sleep(0.04 + rng.uniform(-0.004, 0.004))
            arrivals[k] = rig.write_pack([10 * k, 10 * k + 1], proc_time=0.04)
        time.sleep(0.06)
        for k, t in arrivals.items():
            assert rig.time_of(10 * k) - t < 0.004
            assert 0.015 < rig.time_of(10 * k + 1) - t < 0.026
    finally:
        rig.close()


def test_a_flag_left_set_for_a_pack_already_taken_is_ignored():
    rig = Rig(exp=1)
    try:
        rig.write_pack([1, 2], proc_time=0.01)
        time.sleep(0.05)
        rig.ready.value = True  # set again, but the batch did not change
        time.sleep(0.05)
        assert rig.values() == [1, 2]
        assert not rig.ready.value
    finally:
        rig.close()
