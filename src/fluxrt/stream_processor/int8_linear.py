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
        if rows <= 16:  # _int_mm needs more than 16 rows
            x2 = torch.cat([x2, x2.new_zeros(17 - rows, self.in_features)])
        x_scale = x2.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-8) / 127.0
        x_int8 = (x2.float() / x_scale).round().clamp(-127, 127).to(torch.int8)
        acc = torch._int_mm(x_int8, self.weight_int8.t())
        out = (acc.float() * x_scale * self.weight_scale).to(x.dtype)[:rows]
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], self.out_features)


def quantize_transformer_blocks(transformer: nn.Module) -> int:
    """Swap every nn.Linear inside the double and single blocks for Int8Linear
    (embedders, modulation and the output projection stay bf16). Returns the
    number of layers swapped."""
    swapped = 0
    for blocks in (transformer.transformer_blocks, transformer.single_transformer_blocks):
        for block in blocks:
            for parent in list(block.modules()):
                for name, child in list(parent.named_children()):
                    if type(child) is nn.Linear:
                        setattr(parent, name, Int8Linear(child))
                        swapped += 1
    torch.cuda.empty_cache()
    return swapped
