"""W8A8 INT8 linear layers for the transformer blocks ("int8_linear" config).

Weights: int8 per output channel (symmetric), quantized once at load.
Activations: int8 per token (row), quantized on the fly. The matmul runs on
the INT8 tensor cores (torch._int_mm, int32 accumulate) and is rescaled to
bf16. On an RTX 4090 at the transformer's shapes this is ~3x a bf16 GEMM with
~1.2% relative error per layer; FP8 (e4m3) was 2x at ~3.7% error.

This changes the output numerically — it is opt-in and must be judged against
the run-to-run noise floor (scripts/perf_ab.py compare --noise).
"""

import torch
import torch.nn as nn


# cuBLAS picks an algorithm per matmul shape, and for int8 a never-seen row
# count costs ~3 ms extra per layer on Windows (RTX 5090 Laptop: 3.27 ms vs
# 0.40 ms repeated). The active-row count differs almost every frame, so across
# 100 layers that was a 300-450 ms stall on every new count. Row counts are
# therefore padded up to a multiple of ROW_BUCKET: ~40 shapes in total, primed
# once at startup (prime_row_buckets).
ROW_BUCKET = 64


class Int8Linear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        w = linear.weight.detach().float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        # (N, K) row-major: .t() is the column-major (K, N) operand _int_mm wants
        self.register_buffer("weight_int8", (w / scale).round().clamp(-127, 127).to(torch.int8).contiguous())
        self.register_buffer("weight_scale", scale.reshape(1, -1))  # (1, N) fp32
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, self.in_features)
        rows = x2.shape[0]
        x_scale = x2.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-8) / 127.0
        x_int8 = (x2.float() / x_scale).round().clamp(-127, 127).to(torch.int8)
        # Pad to the next ROW_BUCKET multiple, at least 17 rows more (_int_mm needs
        # more than 16 rows). No size branch: one compiled graph for every count.
        pad = 17 + (-(rows + 17)) % ROW_BUCKET
        x_int8 = torch.cat([x_int8, x_int8.new_zeros(pad, self.in_features)])
        acc = torch._int_mm(x_int8, self.weight_int8.t())[:rows]
        out = (acc.float() * x_scale * self.weight_scale).to(x.dtype)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], self.out_features)


# Output-side projections: their inputs (attention / SwiGLU outputs) carry the
# activation outliers per-token int8 handles worst. "inputs" mode keeps them bf16.
_OUTPUT_SIDE = ("to_out", "to_add_out", "linear_out")


def quantize_transformer_blocks(transformer: nn.Module, mode="all") -> int:
    """Swap nn.Linear layers inside the double and single blocks for Int8Linear
    (embedders, modulation and the output projection always stay bf16).
    mode "all": every block linear; "inputs": only the input-side projections
    (q/k/v, the fused single-block projection, FF in) — ~70% of the GEMM work.
    Returns the number of layers swapped."""
    swapped = 0
    for blocks in (transformer.transformer_blocks, transformer.single_transformer_blocks):
        for block in blocks:
            for module_name, parent in list(block.named_modules()):
                for name, child in list(parent.named_children()):
                    if type(child) is not nn.Linear:
                        continue
                    path = f"{module_name}.{name}"
                    if mode == "inputs" and any(part in path.split(".") for part in _OUTPUT_SIDE):
                        continue
                    setattr(parent, name, Int8Linear(child))
                    swapped += 1
    torch.cuda.empty_cache()
    return swapped


def prime_row_buckets(transformer: nn.Module, max_rows: int) -> int:
    """Run every Int8Linear once at every bucketed row count up to max_rows, so
    cuBLAS has its algorithm for each shape before the first live frame.
    Returns the number of (layer, bucket) pairs touched."""
    layers = [m for m in transformer.modules() if isinstance(m, Int8Linear)]
    touched = 0
    with torch.no_grad():
        for layer in layers:
            weight = layer.weight_int8
            for rows in range(ROW_BUCKET, max_rows + ROW_BUCKET + 1, ROW_BUCKET):
                torch._int_mm(weight.new_zeros(rows, layer.in_features), weight.t())
                touched += 1
    if layers:
        torch.cuda.synchronize()
    return touched
