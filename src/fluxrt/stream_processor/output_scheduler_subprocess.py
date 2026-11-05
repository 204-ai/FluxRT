from multiprocessing import Process, Value
from fluxrt.utils.shared_tensor import SharedTensor
import numpy as np
import time

# How often the idle loop looks for a new pack, and the longest single sleep
# while a frame is held (so a newer pack is noticed this soon).
IDLE_POLL_S = 0.0005
HOLD_POLL_S = 0.001


class OutputSchedulerSubprocess:
    def __init__(
        self,
        config: dict,
        output_batch_shared_tensor_name: str,
        output_shared_tensor_name: str,
        pack_is_ready,
        last_processing_time,
        frame_written=None,
        frame_counter=None,
    ):
        self.config = config
        self.output_batch_shared_tensor_name = output_batch_shared_tensor_name
        self.output_shared_tensor_name = output_shared_tensor_name
        self.pack_is_ready = pack_is_ready
        self.last_processing_time = last_processing_time
        self.frame_written = frame_written
        # Monotonic per-written-frame counter (shared Value). Lets a consumer
        # (run_webrtc's output pump) publish EVERY scheduled frame — including
        # the interpolated in-betweens — instead of sampling at the input rate.
        self.frame_counter = frame_counter

        self.running = Value("b", False)
        self.process = None

        self.interpolation_exp = self.config.get("interpolation_exp", 1)
        self.batch_size = 2**self.interpolation_exp

    def start(self) -> None:
        self.running.value = True
        self.process = Process(target=self.process_main)
        self.process.start()

    def stop(self) -> None:
        from fluxrt.webrtc.proc import reap_process

        self.running.value = False
        if self.process:
            reap_process(self.process)
            self.process = None

    def process_init(self) -> None:
        """
        Called by the internal process
        """
        height = self.config["resolution"]["height"]
        width = self.config["resolution"]["width"]

        if self.config.get("enable_flow_upscaler", False):
            height, width = height * 2, width * 2

        self.output_batch_shared_tensor = SharedTensor(
            (self.batch_size, height, width, 3),
            name=self.output_batch_shared_tensor_name,
        )
        self.output_shared_tensor = SharedTensor(
            (height, width, 3),
            name=self.output_shared_tensor_name,
        )
        # This process's own copy of the pack being shown: the model writes
        # the next pack into shared memory meanwhile (reading batch[i] from
        # shared memory while pacing showed frames of a pack written since).
        self.pack = np.empty(self.output_batch_shared_tensor.shape, dtype=np.uint8)
        self.pack_sequence = -1
        self.queue = []  # frames still to show, oldest first
        self.interval = 0.0  # how long each frame of the current pack is shown
        self.next_due = 0.0

    def take_pack(self) -> bool:
        """Copy the newest pack out of shared memory. False when there is
        nothing new (a flag left set by a pack this process already took)."""
        # Clear the flag BEFORE copying: a pack that lands while this one is
        # shown sets it again and is not lost (the flag used to be cleared
        # after pacing, which dropped it).
        self.pack_is_ready.value = False
        batch = self.output_batch_shared_tensor
        if not batch.read_into(self.pack):
            return True  # no clean copy within the retry budget: show what we got, as before
        if batch.versioned:
            if batch.last_read_sequence == self.pack_sequence:
                return False
            self.pack_sequence = batch.last_read_sequence
        return True

    def publish(self, frame: np.ndarray) -> None:
        self.output_shared_tensor.copy_from(frame)
        if self.frame_written is not None and not self.frame_written.value:
            self.frame_written.value = True
        if self.frame_counter is not None:
            self.frame_counter.value += 1

    def step(self) -> float:
        """One scheduling step. Returns how long to sleep before the next.

        Each pack (interpolated frames, then the generated one) is shown over
        one frame's processing time: its first frame as soon as it arrives,
        each next one `interval` later. A pack that arrives while the
        previous one is still being shown drops the previous pack's
        in-betweens that were not shown yet but keeps its generated frame
        first in line, so every generated frame is shown: that one goes out
        at once, and the new pack follows at the new pack's interval (the
        new frame came sooner than the old interval said)."""
        if self.pack_is_ready.value:
            carry = [self.queue[-1].copy()] if self.queue else []
            if self.take_pack():
                proc_time = min(max(self.last_processing_time.value, 0.001), 1.0)
                self.interval = proc_time / self.batch_size
                self.queue = carry + list(self.pack)
                # no hold before a new pack: its first frame (a pack that
                # comes early is the late one) goes out now, as it always did
                self.next_due = 0.0
        if not self.queue:
            return IDLE_POLL_S
        now = time.perf_counter()
        if now < self.next_due:
            return min(self.next_due - now, HOLD_POLL_S)
        self.publish(self.queue.pop(0))
        self.next_due = now + self.interval
        return 0.0

    def process_main(self) -> None:
        self.process_init()

        while self.running.value:
            wait = self.step()
            if wait > 0:
                time.sleep(wait)
