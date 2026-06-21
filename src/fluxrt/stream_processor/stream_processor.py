import multiprocessing
from multiprocessing import Value
from fluxrt.utils import SharedTensor
from fluxrt.stream_processor.model_inference_subprocess import (
    ModelInferenceSubprocess,
)
from fluxrt.stream_processor.output_scheduler_subprocess import (
    OutputSchedulerSubprocess,
)
import json
import numpy as np


class StreamProcessor:
    def __init__(self, config):
        # `config` is a path to a JSON config OR an already-parsed config dict.
        # The batch render clones the live config and overrides a few keys, so it
        # constructs a StreamProcessor from a dict rather than a file. Copy a passed
        # dict so the batch-mode mutation below can never reach a config a live
        # processor shares (defense-in-depth on top of the caller's deepcopy).
        self.config = dict(config) if isinstance(config, dict) else self.parse_config(config)
        # Batch mode: a dedicated OFFLINE instance that renders exactly one output
        # per submit_frame() — synchronous, no interpolation, no output scheduler.
        # Separate from the live free-run path (which never sets this).
        self.batch_mode = bool(self.config.get("batch_mode", False))
        # Batch is strictly one output per input — force interpolation off before
        # the tensor sizes are derived (both subprocesses re-read this key, so they
        # must agree on the shape).
        if self.batch_mode:
            self.config["interpolation_exp"] = 0
        self.resolution = self.config["resolution"]
        output_batch_size = 2 ** self.config["interpolation_exp"]

        height, width = self.resolution["height"], self.resolution["width"]
        out_height, out_width = height, width
        if self.config.get("enable_flow_upscaler", False):
            out_height, out_width = out_height * 2, out_width * 2

        self.input_shared_tensor = SharedTensor((height, width, 3), create=True)
        self.output_shared_tensor = SharedTensor(
            (out_height, out_width, 3), create=True
        )
        self.output_batch_shared_tensor = SharedTensor(
            (output_batch_size, out_height, out_width, 3), create=True
        )

        self.out_resolution = {"height": out_height, "width": out_width}

        multiprocessing.set_start_method("spawn", force=True)

        self.pack_is_ready = Value("b", False)
        self.last_processing_time = Value("f", 0.0)
        self.frame_written = Value("b", False)

        self.model_inference_subprocess = ModelInferenceSubprocess(
            self.config,
            self.input_shared_tensor.name,
            self.output_batch_shared_tensor.name,
            self.pack_is_ready,
            self.last_processing_time,
        )

        # Batch mode returns frames synchronously via submit_frame(), so the
        # fixed-fps output scheduler (which paces latest_rgb for the live track)
        # is not started.
        if self.batch_mode:
            self.output_scheduler_subprocess = None
        else:
            self.output_scheduler_subprocess = OutputSchedulerSubprocess(
                self.config,
                self.output_batch_shared_tensor.name,
                self.output_shared_tensor.name,
                self.pack_is_ready,
                self.last_processing_time,
                self.frame_written,
            )

    def parse_config(self, config_path: str) -> dict:
        with open(config_path, "r") as file:
            return json.load(file)

    def start(self) -> None:
        self.model_inference_subprocess.start()
        if self.output_scheduler_subprocess is not None:
            self.output_scheduler_subprocess.start()

    def submit_frame(self, frame_rgb: np.ndarray, timeout: float = 300.0) -> np.ndarray:
        """Batch mode only: render ONE input frame synchronously and return its
        output (uint8 RGB). Blocks until the inference subprocess produces it."""
        if not self.batch_mode:
            raise RuntimeError("submit_frame() requires batch_mode=true")
        return self.model_inference_subprocess.submit_frame(frame_rgb, timeout=timeout)

    def worker_alive(self) -> bool:
        """Is the inference subprocess running? Lets a batch caller fail fast if the
        child died (e.g. CUDA OOM during model load) instead of waiting forever."""
        p = getattr(self.model_inference_subprocess, "process", None)
        return bool(p is not None and p.is_alive())

    def get_input_tensor(self) -> SharedTensor:
        return self.input_shared_tensor

    def get_output_tensor(self) -> SharedTensor:
        return self.output_shared_tensor

    def stop(self) -> None:
        # Run every teardown step even if an earlier one raises, so a single
        # failure can't leak subprocesses or shared-memory segments.
        steps = [self.model_inference_subprocess.stop]
        if self.output_scheduler_subprocess is not None:
            steps.append(self.output_scheduler_subprocess.stop)
        steps += [
            self.input_shared_tensor.close_and_unlink,
            self.output_shared_tensor.close_and_unlink,
            self.output_batch_shared_tensor.close_and_unlink,
        ]
        for step in steps:
            try:
                step()
            except Exception as exc:
                print(f"StreamProcessor.stop: {step.__qualname__} failed: {exc}")

    def set_prompt(self, prompt: str) -> None:
        self.model_inference_subprocess.set_param(name="prompt", value=prompt)

    def start_prompt_travel(
        self, target_prompt: str, frames: int = 48, mode: str = "slerp"
    ) -> None:
        """Smoothly morph from the current prompt to target_prompt over
        `frames` generated frames (mode: "slerp" or "lerp")."""
        self.model_inference_subprocess.start_prompt_travel(
            target_prompt, frames, mode
        )

    def set_steps(self, steps: int) -> None:
        self.model_inference_subprocess.set_param(name="steps", value=steps)

    def set_seed(self, seed: int) -> None:
        self.model_inference_subprocess.set_param(name="seed", value=seed)

    def set_param(self, name: str, value) -> None:
        self.model_inference_subprocess.set_param(name=name, value=value)

    def set_reference_image(self, image: np.ndarray | None) -> None:
        if not self.config.get("use_reference_image", False):
            raise ValueError(
                "set_reference_image called but use_reference_image is not enabled in the config"
            )
        self.model_inference_subprocess.set_reference_image(image)

    def set_mask(self, mask: np.ndarray) -> None:
        if self.config.get("mask_calculation_method", "auto") != "manual":
            raise ValueError(
                "set_mask called but mask_calculation_method is not set to manual in the config"
            )
        self.model_inference_subprocess.set_mask(mask)

    def get_resolution(self) -> dict:
        return self.resolution

    def get_out_resolution(self) -> dict:
        return self.out_resolution

    def is_ready(self) -> bool:
        # Batch mode has no output scheduler (frame_written stays False); readiness
        # is the inference subprocess having loaded its models.
        if self.batch_mode:
            return bool(self.model_inference_subprocess.proc_ready.value)
        return bool(self.frame_written.value)

    def get_input_shared_tensor_name(self) -> str:
        return self.input_shared_tensor.name

    def get_output_shared_tensor_name(self) -> str:
        return self.output_shared_tensor.name

    def get_last_processing_time(self) -> float:
        with self.last_processing_time.get_lock():
            return self.last_processing_time.value

    def set_lip_transfer(self, enabled: bool) -> None:
        self.model_inference_subprocess.set_lip_transfer(enabled)

    def enable_quantization(self) -> None:
        self.model_inference_subprocess.enable_quantization()

    def get_reserved_memory(self) -> int:
        """Returns reserved GPU memory in MB."""
        return self.model_inference_subprocess.memory_reserved.value
