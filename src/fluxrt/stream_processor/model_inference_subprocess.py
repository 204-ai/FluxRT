import torch
import time
import cv2
import numpy as np
import json
import os
import signal
from safetensors.torch import load_file
from multiprocessing import Process, Value, Manager
from queue import Empty
from PIL import Image

# ── torch.compile / Dynamo tuning ─────────────────────────────────────────────
# The spatial-cache transformer in transformer_flux2.py specializes Dynamo
# graphs on (a) per-block KV cache indexing (single_block_keys[block_id] for 20
# distinct block_id values) and (b) sparse_mlp_compute() shapes that change
# with the per-frame active-token mask. The default recompile_limit (8) gives
# up partway and falls back to eager mode for the remaining graphs, costing
# ~10-25% throughput. Raising the limit lets every variant compile, then warm
# up once and stay hot.
try:
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.cache_size_limit = 256
except AttributeError:
    # Older torch builds without _dynamo.config — skip silently; nothing breaks.
    pass

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKLFlux2
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM, AutoConfig
from accelerate import init_empty_weights

from fluxrt.stream_processor.interpolation_model import IFNet
from fluxrt.stream_processor.transformer_flux2 import Flux2Transformer2DModel
from fluxrt.utils.shared_tensor import SharedTensor
from fluxrt.stream_processor.pipeline import Flux2KleinPipeline
from fluxrt.stream_processor.update_controller import UpdateController
from fluxrt.stream_processor.postprocessors import (
    BasePostProcessor,
    LivePortraitPostProcessor,
)

from fluxrt.flow_upscaler.upscaler_unet import UpscalerUNet
from fluxrt.flow_upscaler.flow_upscaler_pipeline import FlowUpscalerPipeline
from fluxrt.stream_processor.flux_tiny_vae import DiffusersTAEF2Wrapper


def slerp(a: torch.Tensor, b: torch.Tensor, t: float, eps: float = 1e-6) -> torch.Tensor:
    """
    Spherical-linear interpolation between two prompt-embedding tensors.

    `a` and `b` are treated as single flattened vectors. slerp keeps the
    interpolant's norm roughly constant through the midpoint, which avoids the
    washed-out / low-contrast middle that plain lerp can produce on conditioning
    tensors. Computed in float32 for numerical stability, then cast back to the
    input dtype (the embeddings are bfloat16). Falls back to lerp when the two
    vectors are nearly colinear (sin(theta) -> 0).
    """
    # Compute only the angle/weights in float32 (scalars — cheap, numerically
    # stable). The flattened float32 copies used for the dot product are
    # transient and freed immediately; the final blend is done in the input
    # dtype (bfloat16) to avoid materializing a full-size float32 result every
    # frame on the memory-tight prompt-travel hot path.
    af, bf = a.flatten().float(), b.flatten().float()
    dot = (af @ bf) / (af.norm() * bf.norm() + eps)
    dot = dot.clamp(-1.0, 1.0)
    # Fall back to lerp when nearly colinear in EITHER direction. Near-parallel
    # (dot -> 1) slerp == lerp anyway; near-anti-parallel (dot -> -1) makes
    # sin(theta) -> 0, so the slerp divisor blows up — lerp is the safe path.
    if dot.abs() > 0.9995:
        return torch.lerp(a, b, t)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    w_a = (torch.sin((1.0 - t) * theta) / sin_theta).to(a.dtype)
    w_b = (torch.sin(t * theta) / sin_theta).to(a.dtype)
    return w_a * a + w_b * b


class ModelInferenceSubprocess:
    def __init__(
        self,
        config: dict,
        input_shared_tensor_name: str,
        output_batch_shared_tensor_name: str,
        pack_is_ready,
        last_processing_time,
    ):
        self.running = Value("b", False)
        self.memory_reserved = Value("i", 0)
        self.process = None
        self.config = config
        self.height = self.config["resolution"]["height"]
        self.width = self.config["resolution"]["width"]
        self.resolution = self.config["resolution"]
        self.prompt = self.config["default_prompt"]
        self.logging = self.config.get("logging", True)
        self.input_shared_tensor_name = input_shared_tensor_name
        self.output_batch_shared_tensor_name = output_batch_shared_tensor_name
        self.pack_is_ready = pack_is_ready
        self.last_processing_time = last_processing_time

        self._manager = Manager()
        self.command_queue = self._manager.Queue()
        self.shared_state = self._manager.dict()
        self.interpolation_exp = self.config.get("interpolation_exp", 1)

    def __getstate__(self):
        # The subprocess is spawned via Process(target=self.process_main),
        # which pickles `self`. The SyncManager holds a weakref and is not
        # picklable, so drop it from the child's state. Only the parent calls
        # stop(), where _manager still exists; the picklable queue/dict
        # proxies the child actually uses are kept.
        state = self.__dict__.copy()
        state.pop("_manager", None)
        return state

    def enable_quantization(self):
        """
        Should be called before the subprocess is started.
        """
        self.config["enable_int8_quantization"] = True

    def init_process_state(self):
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.process_state = {
            "prompt": self.config["default_prompt"],
            "steps": self.config["default_steps"],
            "seed": self.config["default_seed"],
        }
        # Active prompt-travel state, or None when no morph is in progress.
        # Populated by _begin_prompt_travel(), advanced in process_main().
        self._travel = None

    def load_models_without_quantization(self):
        device = self.device
        dtype = torch.bfloat16

        models_path = self.config["models_path"]
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            f"{models_path}/scheduler", local_files_only=True, device=device
        )
        self.transformer = Flux2Transformer2DModel.from_pretrained(
            f"{models_path}/transformer", local_files_only=True, device=device
        ).to(dtype)

        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
            f"{models_path}/text_encoder", local_files_only=True
        ).to(device, dtype)
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            f"{models_path}/tokenizer", local_files_only=True, device=device
        )

    def load_quantized_models(self):
        from optimum.quanto import requantize
        from fluxrt.stream_processor.quantized_flux2 import (
            QuantizedFlux2Transformer2DModel,
        )

        models_path = self.config["models_path"]
        int8_models_path = self.config["int8_models_path"]

        qtransformer = QuantizedFlux2Transformer2DModel.from_pretrained(
            int8_models_path, local_files_only=True
        )
        qtransformer.to(device=self.device, dtype=self.dtype)
        self.transformer = qtransformer._wrapped

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            f"{models_path}/scheduler", local_files_only=True, device=self.device
        )

        config = AutoConfig.from_pretrained(
            f"{int8_models_path}/text_encoder", local_files_only=True
        )
        with init_empty_weights():
            text_encoder = Qwen3ForCausalLM(config)

        with open(f"{int8_models_path}/text_encoder/quanto_qmap.json", "r") as f:
            qmap = json.load(f)
        state_dict = load_file(f"{int8_models_path}/text_encoder/model.safetensors")
        requantize(text_encoder, state_dict=state_dict, quantization_map=qmap)
        text_encoder.eval()
        text_encoder.to(self.device, dtype=self.dtype)
        self.text_encoder = text_encoder

        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            f"{int8_models_path}/tokenizer", local_files_only=True
        )

    def load_models(self):
        self.interpolation_model = IFNet()
        self.interpolation_model.load_state_dict(
            load_file("RIFE-safetensors/flownet.safetensors")
        )
        # interpolation model reqires torch.float16, not torch.bfloat16 to avoid pixelation on grid sample layers
        self.interpolation_model.to(self.device, torch.float16)
        self.interpolation_model.eval()

        if self.config.get("enable_int8_quantization", False):
            self.load_quantized_models()
        else:
            self.load_models_without_quantization()

        if self.config.get("enable_flow_upscaler", False):
            self.upscaler_unet = UpscalerUNet()
            state_dict = state_dict = load_file(
                "FlowUpscaler/flow_upscaler.safetensors"
            )
            self.upscaler_unet.load_state_dict(state_dict)
            self.upscaler_unet.to(self.device, self.dtype)
            self.upscaler_pipe = FlowUpscalerPipeline(
                self.upscaler_unet, self.scheduler
            )
        else:
            self.upscaler_pipe = None

        if self.config.get("enable_tiny_vae", False):
            self.vae = DiffusersTAEF2Wrapper(path="taef2/taef2.safetensors").to(
                self.device, self.dtype
            )
        else:
            models_path = self.config["models_path"]
            self.vae = AutoencoderKLFlux2.from_pretrained(
                f"{models_path}/vae", local_files_only=True, device=self.device
            ).to(self.dtype)

        if self.config.get("compile_models", False):
            self.transformer = torch.compile(
                self.transformer,
            )
            self.vae = torch.compile(
                self.vae,
            )
            self.interpolation_model = torch.compile(
                self.interpolation_model,
            )

        reference_image_seq_len = None
        if self.config.get("use_reference_image", False):
            reference_image_res = self.config["reference_image_resolution"]
            reference_image_seq_len = (reference_image_res["width"] // 16) * (
                reference_image_res["height"] // 16
            )

        self.update_controller = UpdateController(
            self.config,
            self.height,
            self.width,
            compression_ratio=16,
            reference_image_seq_len=reference_image_seq_len,
        )

        self.pipe = Flux2KleinPipeline(
            scheduler=self.scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            update_controller=self.update_controller,
            subprocess_config=self.config,
            upscaler_pipeline=self.upscaler_pipe,
        )
        self.pipe.to(self.device)

        if self.config.get("use_lora", False):
            self.pipe.load_lora_weights(self.config.get("lora_weights_path", ""))

        self.lip_processor: BasePostProcessor | None = None
        self.lip_active = False
        lp_cfg = self.config.get("lip_transfer", {})
        if lp_cfg.get("enable", False):
            self.lip_processor = LivePortraitPostProcessor(
                models_dir=lp_cfg["models_dir"]
            )

    def update_prompt_embeds(self, prompt):
        self.prompt_embeds, text_ids = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )
        # A direct prompt set cancels any in-progress morph.
        self._travel = None
        self.update_controller.reset_cache()

    def _begin_prompt_travel(self, payload: dict) -> None:
        """
        Worker-side handler: pre-encode the target prompt ONCE and arm the
        travel state. Encoding (the text-encoder forward) is the expensive part,
        so it must not run per frame — only the cheap blend does.
        """
        target_prompt = payload["prompt"]
        frames = max(1, int(payload.get("frames", 48)))
        mode = payload.get("mode", "slerp")
        target_embeds, _ = self.pipe.encode_prompt(
            prompt=target_prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )
        # Full-frame execute (the expensive spatial-cache-disabling dense pass)
        # is applied only every `stride` frames during the morph instead of
        # every frame; >=1, default 2 (~halves the dense-execute cost). Override
        # with config "prompt_travel_full_execute_every" (1 = old behaviour).
        stride = max(1, int(self.config.get("prompt_travel_full_execute_every", 2)))
        self._travel = {
            "src": self.prompt_embeds,
            "tgt": target_embeds,
            "n": frames,
            "i": 0,
            "mode": mode,
            "prompt": target_prompt,
            "stride": stride,
        }

    def _advance_prompt_travel(self) -> None:
        """
        Called once per generated frame. Advances an in-progress morph by one
        step: blends src->tgt embeddings so the new conditioning reaches the
        frame. To keep the morph cheap, the EXPENSIVE full-frame execute (which
        disables the spatial sparse cache and recomputes every image token) is
        applied only every `stride` frames; the cheap text-K/V recompute runs
        every frame so MOVING regions keep morphing smoothly, while STATIC
        regions catch up on the strided full-execute frames.
        """
        tv = self._travel
        if tv is None:
            return

        # Advance first so the morph renders exactly tv["n"] frames, every one of
        # them showing progress: the first frame is at t=1/n (not t=0, which
        # would be the unchanged source) and the last is at t=n/n=1.0 (target).
        tv["i"] += 1
        t = tv["i"] / tv["n"]
        if tv["mode"] == "lerp":
            self.prompt_embeds = torch.lerp(tv["src"], tv["tgt"], t)
        else:
            self.prompt_embeds = slerp(tv["src"], tv["tgt"], t)

        # Recompute text K/V every frame (cheap — 512 text tokens). Executing
        # image tokens (the moving regions the spatial cache doesn't skip) cross-
        # attend to the fresh K/V and pick up the new conditioning, so motion
        # morphs smoothly every frame without a full dense execute.
        self.update_controller.text_is_valid = False

        # Full-frame execute (requires_reset -> ALL image tokens recompute) is
        # the costly part that also drags STATIC/cached tokens onto the new
        # conditioning. Stride it: the first frame, every `stride`th frame, and
        # the final frame (which must land the exact target everywhere). Caps the
        # dense-execute cost at ~1/stride of the per-frame version.
        last = tv["i"] >= tv["n"]
        if last or tv["i"] == 1 or tv["i"] % tv["stride"] == 0:
            self.update_controller.requires_reset = True

        if last:
            # Final frame was rendered at t=1.0 above; snap exactly onto the
            # target embeds and update tracked prompt state.
            self.prompt_embeds = tv["tgt"]
            self.process_state["prompt"] = tv["prompt"]
            self._travel = None

    def init_shared_tensors(self):
        height, width = self.resolution["height"], self.resolution["width"]
        out_height, out_width = height, width

        if self.config.get("enable_flow_upscaler", False):
            out_height, out_width = out_height * 2, out_width * 2

        self.input_shared_tensor = SharedTensor(
            (height, width, 3),
            name=self.input_shared_tensor_name,
        )

        # All interpolated then one original
        output_batch_size = 2**self.interpolation_exp
        self.output_batch_shared_tensor = SharedTensor(
            (output_batch_size, out_height, out_width, 3),
            name=self.output_batch_shared_tensor_name,
        )

    def process_init(self):
        """
        Initializes all resources required by the inference subprocess.
        """
        self.init_process_state()
        self.init_shared_tensors()
        self.load_models()
        self.update_prompt_embeds(self.process_state["prompt"])
        self.previous_frame = None

        if self.config.get("use_reference_image", False):
            image = cv2.imread(self.config.get("reference_image_path", ""))
            resolution = self.config.get("reference_image_resolution")
            if image is None:
                image = np.zeros(
                    (resolution["height"], resolution["width"], 3), dtype=np.uint8
                )
                print(
                    "Warning: use_reference_image is set to true but no valid reference_image_path is provided."
                )
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (resolution["width"], resolution["height"]))
            self.reference_image = Image.fromarray(image)

        target_fps = self.config.get("target_fps", None)
        self.target_base_processing_time = None
        if target_fps is not None:
            target_base_fps = target_fps / (2**self.interpolation_exp)
            self.target_base_processing_time = 1 / target_base_fps

    def start(self):
        self.running.value = True
        self.process = Process(target=self.process_main)
        self.process.start()

    def stop(self):
        self.running.value = False
        if self.process:
            # Graceful first: let process_main observe running=False and exit
            # its loop. Escalate to terminate/kill if it is wedged in CUDA,
            # torch.compile, or a blocking call so Ctrl+C never hangs.
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
            if self.process.is_alive():
                # SIGKILL didn't reap within 2s — typically a child stuck in an
                # uninterruptible CUDA driver syscall (D state). Re-signal and
                # DETACH it from multiprocessing's _children so the interpreter-
                # exit atexit join (which has NO timeout) can never block on this
                # pid and leave an orphaned GPU process needing a manual kill.
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
        # Tear down the Manager process spawned in __init__; otherwise it
        # lingers as an orphan after the main process exits.
        if getattr(self, "_manager", None) is not None:
            try:
                self._manager.shutdown()
            except Exception:
                pass
            self._manager = None

    def set_param(self, name: str, value) -> None:
        self.command_queue.put(("set_param", (name, value)))

    def set_reference_image(self, image: np.ndarray | None) -> None:
        """
        Update the reference image on the fly.
        image: numpy uint8 RGB array
        Only valid when use_reference_image is true in config.
        """
        if not self.config.get("use_reference_image", False):
            raise ValueError(
                "set_reference_image called but use_reference_image is not enabled in the stream processor config"
            )
        self.command_queue.put(("set_reference_image", image))

    def set_mask(self, mask) -> None:
        """
        Update the mask on the fly.
        mask: numpy uint8 array of shape (h // compression_ratio, w // compression_ratio).
        Only valid when mask_calculation_method is set to manual in config.
        """
        if self.config.get("mask_calculation_method", "auto") != "manual":
            raise ValueError(
                "set_mask called but mask_calculation_method is not set to manual in the config"
            )
        self.command_queue.put(("set_mask", mask))

    def set_lip_transfer(self, enabled: bool) -> None:
        self.command_queue.put(("set_lip_transfer", enabled))

    def start_prompt_travel(
        self, target_prompt: str, frames: int = 48, mode: str = "slerp"
    ) -> None:
        """
        Smoothly interpolate the conditioning from the current prompt to
        `target_prompt` over `frames` generated frames. `mode` is "slerp" or
        "lerp". Enqueues onto the command queue; handled in the inference
        subprocess by _begin_prompt_travel / _advance_prompt_travel.
        """
        self.command_queue.put(
            (
                "start_prompt_travel",
                {"prompt": target_prompt, "frames": int(frames), "mode": mode},
            )
        )

    def update_process_state(self) -> None:
        """
        Called by the internal process
        """
        try:
            while True:
                cmd, payload = self.command_queue.get_nowait()
                if cmd == "set_param":
                    name, value = payload
                    self.process_state[name] = value
                    if name == "prompt":
                        self.update_prompt_embeds(value)
                elif cmd == "set_reference_image":
                    image = payload  # numpy uint8 RGB array or None
                    resolution = self.config["reference_image_resolution"]
                    if image is not None:
                        image = cv2.resize(
                            image, (resolution["width"], resolution["height"])
                        )
                        self.reference_image = Image.fromarray(image)
                    else:
                        self.reference_image = Image.fromarray(
                            np.zeros(
                                (resolution["height"], resolution["width"], 3),
                                dtype=np.uint8,
                            )
                        )
                    self.update_controller.reset_cache()

                elif cmd == "set_mask":
                    mask = payload  # numpy uint8 array of shape (h // compression_ratio, w // compression_ratio)
                    mask_tensor = (
                        torch.from_numpy(mask)
                        .unsqueeze(0)
                        .to(self.update_controller.device)
                    )
                    self.update_controller.set_mask(mask_tensor)

                elif cmd == "set_lip_transfer":
                    self.lip_active = payload

                elif cmd == "start_prompt_travel":
                    self._begin_prompt_travel(payload)

        except Empty:
            pass

    def receive_frame(self):
        """
        Reads frame from input shared memory, converts to RGB float16 GPU tensors.
        """
        frame = self.input_shared_tensor.to_numpy()
        frame_gpu = (
            torch.from_numpy(frame)
            .to(self.device)
            .to(torch.float16)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .div(255)
        )
        return frame_gpu

    def interpolate_frames(self, frame):
        """
        Takes one new generated frame (torch tensor, RGB, on GPU, float16)
        Interpolates according to interpolation_exp times.
        Batches to [interpolated frames, new frame].
        """
        if self.previous_frame is None:
            self.previous_frame = frame

        if self.interpolation_exp == 0:
            frames_out = frame
        else:
            frames = torch.cat([self.previous_frame, frame], dim=0)
            with torch.no_grad():
                for _ in range(self.interpolation_exp):
                    B = frames.size(0)
                    prevs = frames[:-1]
                    nexts = frames[1:]
                    mids = self.interpolation_model(torch.cat([prevs, nexts], dim=1))
                    H, W = frames.shape[2:]
                    new_frames = torch.empty(
                        2 * B - 1, 3, H, W, device=frames.device, dtype=frames.dtype
                    )
                    new_frames[0::2] = frames
                    new_frames[1::2] = mids
                    frames = new_frames
            frames_out = frames[1:]

        frames_cpu = (
            frames_out.mul(255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )

        self.previous_frame = frame

        return frames_cpu[..., ::-1]

    def send_frames(self, frames):
        self.output_batch_shared_tensor.copy_from(frames)

    def sync_fps_and_send(self, prev_time, frames):
        now = time.time()
        processing_time = now - prev_time

        if self.target_base_processing_time is not None:
            sleep_time = max(0, self.target_base_processing_time - processing_time)
            time.sleep(sleep_time)
            now = time.time()

        processing_time = now - prev_time

        self.last_processing_time.value = processing_time
        self.send_frames(frames)
        self.pack_is_ready.value = True
        self.memory_reserved.value = torch.cuda.memory_reserved() // (1024 * 1024)

        if self.logging:
            print(
                f"base fps: {(1 / processing_time):.2f}, interpolated fps: {(1 / processing_time * 2**self.interpolation_exp):.2f}"
            )
        return now

    def process_frame_with_pipeline(self, frame):
        """
        Takes frame as np uint8 RGB array
        Returns frame as np uint8 RGB array
        """
        input_frame = Image.fromarray(frame)

        reference_list = [input_frame]
        if self.config["use_reference_image"]:
            reference_list.append(self.reference_image)

        out = self.pipe(
            prompt_embeds=self.prompt_embeds,
            image=reference_list,
            height=self.resolution["height"],
            width=self.resolution["width"],
            guidance_scale=1.0,
            num_inference_steps=self.process_state["steps"],
            num_images_per_prompt=1,
            generator=torch.Generator(device=self.device).manual_seed(
                self.process_state["seed"]
            ),
            output_type="np",
        )
        out_image = out.images[0]
        out_image = out_image * 255
        out_image = out_image.astype(np.uint8)
        return out_image

    def convert_np_to_torch(self, frame):
        frame = (
            torch.from_numpy(frame)
            .to(self.device)
            .to(torch.float16)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .div(255)
        )
        return frame

    def process_main(self):
        # Ignore SIGINT in the child. With the "spawn" start method the child
        # shares the parent's process group, so Ctrl+C is delivered here too;
        # the default handler would raise KeyboardInterrupt mid-CUDA and can
        # leave a half-torn context (D state) that resists kill. Shut down only
        # via running.value (parent flips it in stop()) / parent terminate+kill.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.process_init()
        prev_time = time.time()
        while self.running.value:
            self.update_process_state()
            self._advance_prompt_travel()
            original_frame = self.input_shared_tensor.to_numpy()
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            frame = self.process_frame_with_pipeline(original_frame)
            if self.lip_processor is not None and self.lip_active:
                # Note: we are getting the latest input frame again after flux processing to reduce latency.
                original_frame = self.input_shared_tensor.to_numpy()
                original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                frame = self.lip_processor.process(frame, original_frame)
            frame = self.convert_np_to_torch(frame)
            frames = self.interpolate_frames(frame)
            prev_time = self.sync_fps_and_send(prev_time, frames)
