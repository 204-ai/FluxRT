"""Fixed-shape conv stages as TensorRT engines ("conv_backend": "tensorrt").

On the RTX 5090 laptop a TensorRT fp16 engine runs the flow upscaler's UNet in
6.1 ms (torch, compiled: 9.7), RIFE at 1152x640 in 6.3 ms (11.6) and the TAEF2
encoder in 1.2 ms (3.1). The engines compute in fp16 with TensorRT's own
kernels, so the output differs at rounding level from the torch path: opt-in,
to be judged against the run-to-run noise floor like any other such change.

Needs `tensorrt` and `onnx` in the environment. An engine is built once per
stage and input shape (10-50 s) and kept on disk; anything that goes wrong
falls back to the torch module.
"""

import hashlib
import os
import time

import torch
import torch.nn as nn

CACHE_DIR = os.environ.get("FLUXRT_TRT_CACHE", os.path.expanduser("~/.cache/fluxrt/tensorrt"))


def available() -> bool:
    try:
        import onnx  # noqa: F401
        import tensorrt  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


class _Positional(nn.Module):
    """The module called with its tensors in a fixed positional order (ONNX export)."""

    def __init__(self, module, names):
        super().__init__()
        self.module, self.names = module, names

    def forward(self, *tensors):
        args = [t for name, t in zip(self.names, tensors) if isinstance(name, int)]
        kwargs = {name: t for name, t in zip(self.names, tensors) if not isinstance(name, int)}
        return self.module(*args, **kwargs)


class TrtStage(nn.Module):
    """`module` (a plain, uncompiled nn.Module with tensor inputs and one tensor
    output) run as a TensorRT fp16 engine, one per set of input shapes.
    The result comes back in the dtype of the first input."""

    def __init__(self, module: nn.Module, name: str):
        super().__init__()
        self.module = module
        self.name = name
        self.engines = {}  # shapes -> (context, engine, static inputs, output) | None (torch fallback)
        self._weights = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def forward(self, *args, **kwargs):
        inputs = dict(enumerate(args)) | kwargs
        if not inputs or not all(torch.is_tensor(v) and v.is_cuda for v in inputs.values()):
            return self.module(*args, **kwargs)
        key = tuple((name, tuple(v.shape)) for name, v in inputs.items())
        if key not in self.engines:
            self.engines[key] = self._load(key, inputs)
        entry = self.engines[key]
        if entry is None:
            return self.module(*args, **kwargs)
        context, _, static, output = entry
        for name, tensor in static.items():
            tensor.copy_(inputs[name])
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        # a new tensor: the engine's output buffer is overwritten by the next call
        return output.to(next(iter(inputs.values())).dtype, copy=True)

    def _fingerprint(self) -> str:
        if self._weights is None:
            digest = hashlib.sha1()
            for name, p in self.module.state_dict().items():
                digest.update(f"{name}{tuple(p.shape)}{float(p.float().abs().sum()):.6e}".encode())
            self._weights = digest.hexdigest()[:12]
        return self._weights

    def _load(self, key, inputs):
        try:
            import tensorrt as trt

            shapes = "_".join("x".join(map(str, shape)) for _, shape in key)
            gpu = torch.cuda.get_device_name(0).replace(" ", "")
            path = os.path.join(CACHE_DIR, f"{self.name}-{shapes}-{self._fingerprint()}-{gpu}-trt{trt.__version__}.plan")
            logger = trt.Logger(trt.Logger.ERROR)
            if not os.path.exists(path):
                start = time.time()
                self._build(trt, logger, path, inputs)
                print(f"tensorrt: built {self.name} {shapes} in {time.time() - start:.0f} s -> {path}")
            with open(path, "rb") as f:
                engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()
            static, output = {}, None
            names = list(inputs)
            for i in range(engine.num_io_tensors):
                tensor_name = engine.get_tensor_name(i)
                shape = tuple(context.get_tensor_shape(tensor_name))
                tensor = torch.empty(shape, device="cuda", dtype=torch.float16)
                if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    static[names[int(tensor_name[2:])]] = tensor  # exported as "in0", "in1", ...
                else:
                    output = tensor
                context.set_tensor_address(tensor_name, tensor.data_ptr())
            return context, engine, static, output
        except Exception as error:  # noqa: BLE001 — any failure means: use the torch module
            print(f"tensorrt: {self.name} falls back to torch ({type(error).__name__}: {str(error).splitlines()[0][:200]})")
            return None

    def _build(self, trt, logger, path, inputs):
        import copy

        os.makedirs(CACHE_DIR, exist_ok=True)
        onnx_path = path[: -len(".plan")] + ".onnx"
        names = list(inputs)
        # TensorRT networks are strongly typed: the precision comes from the
        # exported model, so export an fp16 copy.
        model = _Positional(copy.deepcopy(self.module), names).half().eval()
        example = tuple(inputs[name].half() for name in names)
        with torch.no_grad():
            torch.onnx.export(
                model,
                example,
                onnx_path,
                input_names=[f"in{i}" for i in range(len(names))],
                output_names=["out"],
                opset_version=17,
                dynamo=False,
            )
        del model
        builder = trt.Builder(logger)
        flag = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
        network = builder.create_network(1 << int(flag) if flag is not None else 0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(onnx_path):
            raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        plan = builder.build_serialized_network(network, builder.create_builder_config())
        if plan is None:
            raise RuntimeError("engine build failed")
        with open(path, "wb") as f:
            f.write(plan)
        os.remove(onnx_path)
