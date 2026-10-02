"""Recording and replaying CUDA graphs.

A recorded graph replays the same kernels on the same tensors with one launch
instead of one per kernel: bit-identical to the call it was recorded from,
without the per-launch CPU cost (large on Windows, where every launch goes
through the OS GPU scheduler). Used for the transformer step
(transformer_flux2.StepGraphs) and the fixed-shape conv stages
(RecordedModule).

Nothing may wait for the GPU while a graph is recorded (a block compiling, a
Triton kernel autotuning, a cuDNN plan being built), so callers record a call
only after running it normally on the very objects the recording uses.
"""

import torch
import torch.nn as nn

# A failed recording leaves its call on the normal path; after this many
# failures nothing more is recorded this session (recorded graphs keep
# replaying).
MAX_FAILURES = 3
failures = 0


def recording() -> bool:
    return failures < MAX_FAILURES


def record(fn, pool=None, what: str = "call"):
    """Record fn() as a CUDA graph (recording executes nothing).
    Returns (graph, fn's result), or None if the recording failed."""
    global failures
    stream = torch.cuda.current_stream()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, pool=pool):
            result = fn()
    except RuntimeError as error:
        # torch.cuda.graph leaves its capture stream current when a recording
        # fails, and CUDA reports the failure once more on the next kernel
        # launch: switch back, and take that report here.
        torch.cuda.set_stream(stream)
        try:
            torch.zeros(1, device="cuda").add_(1)
        except RuntimeError:
            pass
        failures += 1
        print(f"cuda graphs: recording a {what} failed, it keeps running normally ({str(error).splitlines()[0]})")
        return None
    return graph, result


class RecordedModule(nn.Module):
    """A module with fixed input shapes (VAE encoder / decoder, upscaler UNet)
    replayed from a CUDA graph, one per set of input shapes.

    The returned tensor is the graph's own and is overwritten by the next call
    with the same shapes: use it before calling again."""

    MAX_GRAPHS = 8

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        # shapes -> "seen" | "normal" (recording failed) | (graph, output, static inputs)
        self.graphs = {}

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def forward(self, *args, **kwargs):
        inputs = dict(enumerate(args)) | kwargs
        if not inputs or not all(torch.is_tensor(v) and v.is_cuda for v in inputs.values()):
            return self.module(*args, **kwargs)
        key = tuple((name, tuple(v.shape), v.dtype) for name, v in inputs.items())
        entry = self.graphs.get(key)
        if entry is None:
            if recording() and len(self.graphs) < self.MAX_GRAPHS:
                self.graphs[key] = "seen"
            return self.module(*args, **kwargs)
        if entry == "seen" and recording():
            static = {name: v.clone() for name, v in inputs.items()}
            call = lambda: self.module(
                *(static[i] for i in range(len(args))), **{name: static[name] for name in kwargs}
            )
            result = call()
            recorded = record(call, what="conv stage")
            self.graphs[key] = "normal" if recorded is None else (*recorded, static)
            return result
        if isinstance(entry, str):
            return self.module(*args, **kwargs)
        graph, output, static = entry
        for name, v in inputs.items():
            static[name].copy_(v)
        graph.replay()
        return output
