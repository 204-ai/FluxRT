"""Memory-traffic and FLOP model of one FLUX.2-klein-4B transformer step in FluxRT at 576x320:
the current code path (full-length residual stream, per-op gather/scatter) against a compact
residual stream (active rows only inside the step). Rough: counts tensor reads/writes of the
main ops. See docs/research-2026-11.md section 1. Run: python scripts/perf_model_traffic.py"""
D, H, HD = 3072, 24, 128
FF = int(3 * D)            # 9216 (SwiGLU: fused in-proj 2*FF)
TXT, IMG = 512, 720 * 2    # text tokens, noisy + condition image tokens
REF = 0                    # reference tokens (256 when a reference image is set)
SEQ = TXT + IMG + REF
N_DOUBLE, N_SINGLE = 5, 20
B2 = 2                     # bytes per bf16 element

def mb(x): return x / 1e6

def step(active_img, active_txt=0, compact=False):
    """Returns (weight bytes, activation bytes, GFLOP) for one step."""
    a = active_img + active_txt
    w = 0.0; act = 0.0; flop = 0.0
    def gemm(rows, k, n, int8=True):
        nonlocal w, act, flop
        w += k * n * (1 if int8 else 2)
        act += rows * k * B2 + rows * n * B2 + (rows * k if int8 else 0)  # read in, write out, int8 copy
        flop += 2 * rows * k * n
    def elementwise(rows, passes):  # full-length (rows) bf16 read+write passes over D
        nonlocal act
        act += passes * rows * D * B2 * 2
    res_rows = a if compact else SEQ
    # double blocks (img and txt streams)
    for _ in range(N_DOUBLE):
        elementwise(res_rows, 2)                       # norm1 + modulation (img)
        elementwise(TXT if not compact else active_txt, 2)  # txt stream norm1 + mod
        for proj in (D, D, D):                          # q, k, v of active img rows
            gemm(active_img, D, proj)
        for proj in (D, D, D):
            gemm(active_txt, D, proj)
        # attention: queries a, keys SEQ (KV cache read), 24 heads
        act += (2 * SEQ * D * B2) + (a * D * B2 * 2)
        flop += 4 * a * SEQ * D
        gemm(active_img, D, D); gemm(active_txt, D, D)  # out projections
        if not compact: act += (SEQ + TXT) * D * B2      # zero-filled full-length outputs
        elementwise(res_rows, 3)                        # gate*attn + residual, norm2 + mod
        gemm(active_img, D, 2 * FF); gemm(active_img, FF, D)
        gemm(active_txt, D, 2 * FF); gemm(active_txt, FF, D)
        if not compact: act += (SEQ + TXT) * D * B2      # zero-filled mlp outputs
        elementwise(res_rows, 1)                        # gate*ff + residual
    # single blocks (joint sequence)
    for _ in range(N_SINGLE):
        elementwise(res_rows, 2)                        # norm + modulation
        gemm(a, D, 3 * D + 2 * FF)                      # fused qkv + mlp in
        act += (2 * SEQ * D * B2) + (a * D * B2 * 2)    # attention over the KV cache
        flop += 4 * a * SEQ * D
        gemm(a, D + FF, D)                              # fused out
        if not compact: act += SEQ * D * B2             # zero-filled output
        elementwise(res_rows, 1)                        # gate + residual
    gemm(active_img, D, 128)                            # proj_out (active rows)
    return w, act, flop / 1e9

for label, rows in (("steady, ~15% changed", (320, 0)), ("busy, ~50% changed", (704, 0)), ("prompt travel (text active)", (320, 512)), ("full frame", (IMG, TXT))):
    for compact in (False, True):
        w, act, gf = step(*rows, compact=compact)
        print(f"{label:30s} {'compact' if compact else 'current':8s} weights {mb(w):6.0f} MB  activations {mb(act):6.0f} MB  {gf:6.0f} GFLOP"
              f"  | 4090 est: mem {(w+act)/1.0e12*1e3:5.1f} ms, int8 compute @50% {gf/330e3*1e3:5.1f} ms")
