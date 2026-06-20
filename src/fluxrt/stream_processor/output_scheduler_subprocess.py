from multiprocessing import Process, Value
from fluxrt.utils.shared_tensor import SharedTensor
import os
import signal
import time


class OutputSchedulerSubprocess:
    def __init__(
        self,
        config: dict,
        output_batch_shared_tensor_name: str,
        output_shared_tensor_name: str,
        pack_is_ready,
        last_processing_time,
        frame_written=None,
    ):
        self.config = config
        self.output_batch_shared_tensor_name = output_batch_shared_tensor_name
        self.output_shared_tensor_name = output_shared_tensor_name
        self.pack_is_ready = pack_is_ready
        self.last_processing_time = last_processing_time
        self.frame_written = frame_written

        self.running = Value("b", False)
        self.process = None

        self.interpolation_exp = self.config.get("interpolation_exp", 1)
        self.batch_size = 2**self.interpolation_exp

    def start(self) -> None:
        self.running.value = True
        self.process = Process(target=self.process_main)
        self.process.start()

    def stop(self) -> None:
        self.running.value = False
        if self.process:
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
            if self.process.is_alive():
                # Detach a SIGKILL survivor from multiprocessing's _children so
                # the no-timeout atexit join can't hang the parent on exit.
                try:
                    os.kill(self.process.pid, signal.SIGKILL)
                except Exception:
                    pass
                try:
                    import multiprocessing.process as _mpp

                    _mpp._children.discard(self.process)
                except Exception:
                    pass
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

    def process_main(self) -> None:
        # See ModelInferenceSubprocess.process_main: ignore SIGINT in the child
        # so Ctrl+C doesn't KeyboardInterrupt it mid-loop; exit via running.value.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.process_init()

        while self.running.value:
            if not self.pack_is_ready.value:
                continue

            proc_time = min(max(self.last_processing_time.value, 0.001), 1.0)
            sleep_interval = proc_time / self.batch_size

            for i in range(self.batch_size):
                self.output_shared_tensor.copy_from(
                    self.output_batch_shared_tensor.array[i]
                )
                if self.frame_written is not None and not self.frame_written.value:
                    self.frame_written.value = True
                if i < self.batch_size - 1:
                    time.sleep(sleep_interval)

            self.pack_is_ready.value = False
