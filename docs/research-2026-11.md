# Research: where FluxRT can still get faster, and what else exists (November 2026)

What was found by reading the code (FluxRT's patched diffusers copies and
diffusers 0.42.1 itself) and by four research sweeps of GitHub, the
model hubs and the real-time generative-video communities on 2026-11-05.
Written for the next round of performance work after `docs/perf-notes.md`,
which records what has been measured and dropped already; nothing here
repeats a dropped idea without new evidence.

Reachability note: the research ran from a cloud sandbox that could reach
GitHub, git and web-search snippets, but not huggingface.co, arxiv.org,
bfl.ai, daydream.live, x.com, reddit.com or the Banodoco Discord corpus.
Claims that rest on search snippets alone are marked *[snippet]*; what
could not be confirmed is marked *[unverified]*; anything after mid-2026 is
marked *[new]* because it postdates the researcher's training data.

## Summary

1. **The transformer step is bandwidth-bound at steady state.** One step
   streams 3.7 GB of int8 weights and ~4.2 GB of activations; two steps =
   ~15 GB per frame, 15 ms on the 4090's 1 TB/s before any compute. The
   two levers that scale with every frame are 4-bit weights (halve the
   weight stream) and fewer full-length activation passes (section 2.1: a
   compact residual stream removes ~2.2 GB per step, est. 3-5 ms per frame
   on the 4090, more on the laptop, bit-identical).
2. **Nothing replaces FLUX.2-klein-4B in its role today.** FLUX.3 Image
   Klein *[new]* (diffusers 0.42, 23 Oct 2026) is 5.5 B parameters with a
   4 B Qwen3-VL text encoder, distilled for 16 steps, ~4 s per megapixel on
   a 5090 *[snippet]*: a benchmark candidate only if a few-step distillation
   appears. Its architecture (reference and text tokens at timestep 0 with
   their own blocks) is friendlier to caching than FLUX.2's.
3. **The real alternatives are streaming video editors**, a class that did
   not exist when FluxRT was written: NVIDIA SANA-Streaming (2 B, 24 fps at
   1280x704 on one RTX 5090, instruction-only, no reference image) and JD's
   JoyAI-Video-Edit (16 B + 7 B, 24 fps at 840x480 on one 5090, with a
   reference image: the only open model that does virtual try-on live),
   both Linux, ~1 s glass-to-glass against FluxRT's 0.2-0.3 s. A
   side-by-side on the team's GPUs is the highest-value experiment.
4. **Upstream FluxRT is frozen** (last commit 13 June 2026, maintainer
   silent since 4 June, the fork's PRs closed unanswered). The fork
   contains every upstream commit and is the most advanced FluxRT there is.
5. **GPU video codecs need no new dependency.** The pip PyAV wheel FluxRT
   already installs ships `h264_nvenc` / `hevc_nvenc` / `av1_nvenc` and the
   NVDEC decoders (verified on the installed 17.1.0; Linux and Windows
   builds, driver >= 570). Encoding the output on NVENC inside aiortc's
   existing path is a one- to two-day change that removes the 10-14 ms
   software VP8 encode per viewer and most of the server's CPU.
6. On consumer Blackwell INT8 and FP8 have the same tensor-core peak (838
   TOPS on the 5090) and FP4 is the only 2x step; the fork's "FP8 slower
   than INT8" result is a 4090 result (where FP8's peak is half INT8's) and
   should be re-measured on the laptop. 4-bit weights (SVDQuant / NVFP4)
   are the one model-side change that moves the laptop's GEMM time
   (24 -> ~13-16 ms per frame), at a quality cost far above INT8's, and
   without upstream nunchaku support for FLUX.2 yet.
7. Ranked list with effort and evidence: section 8.

## 1. What a frame costs, from the code

FLUX.2-klein-4B as FluxRT runs it: 5 double-stream blocks + 20 single-stream
blocks, 24 heads x 128 = 3072 wide, SwiGLU feed-forward x3 (a fused 2x9216
in-projection), single blocks with one fused qkv+MLP projection
(3072 -> 27648). About 3.7 B parameters in the blocks. The sequence at
576x320 is 512 text tokens + 720 noisy latent tokens + 720 condition-image
tokens (+256 with a reference image) = 1952 (2208). Per frame: TAEF2/VAE
encode of the input (1.2-2.1 ms), change mask (1 ms), 2 Euler steps,
optionally the 2x latent upscaler (1 step), TAEF2/VAE decode, RIFE x2.

The spatial cache keeps, per denoising step, a `SpatialCache` with every
block's K/V and the output cache. Only "active" rows run (changed image
tokens after dilation; text rows only while the prompt changes; reference
rows once); inactive rows' K/V come from the cache. Row counts are bucketed
to multiples of 64; steady state is ~320 active rows, motion ~700, prompt
travel +512 text rows.

A memory-traffic model of one step (`scripts/perf_model_traffic.py`; counts
the reads and writes of the main ops, int8 weights, bf16 activations,
int8 compute at 50 % of the 4090's 660 TOPS):

| Active rows | Residual stream | Weights | Activations | GFLOP | Memory time (4090) | Compute (4090) |
|---|---|---:|---:|---:|---:|---:|
| 320 (steady) | current, full length | 3.7 GB | 4.2 GB | 2155 | 7.9 ms | 6.5 ms |
| 320 | compact (section 2.1) | 3.7 GB | 2.0 GB | 2155 | 5.7 ms | 6.5 ms |
| 704 (motion) | current | 3.7 GB | 5.4 GB | 4741 | 9.1 ms | 14.4 ms |
| 704 | compact | 3.7 GB | 3.6 GB | 4741 | 7.3 ms | 14.4 ms |
| 320 + 512 text (travel) | current | 3.7 GB | 5.8 GB | 5603 | 9.5 ms | 17.0 ms |

Reading: at steady state a step is bound by bandwidth, and the weight
stream alone (3.7 GB per step, 7.4 GB per frame) is a floor of ~7.4 ms per
frame on a 4090 and ~8.2 ms on the laptop's 896 GB/s, whatever else
improves. The laptop at 110 W is also compute-bound (measured 12 ms of GEMM
per step, `perf-notes.md`). The measured all-in frame on the 4090 is
32.5 ms: the transformer's two steps are within 2x of this floor already.

## 2. Code-level findings in FluxRT's transformer

### 2.1 Compact residual stream (bit-identical; est. -3 to -5 ms per frame on the 4090, more on the laptop)

Projections, MLPs and attention queries already run on active rows only,
but the residual stream stays full length: `Flux2SingleTransformerBlock.forward`
and `Flux2TransformerBlock.forward` run LayerNorm, modulation and
`gate * out + residual` over all 1952 rows (12 MB per bf16 tensor), and
every sparse op zero-fills a full-length output buffer and `index_copy_`s
its rows into it (`sparse_mlp_compute`, `_flux2_parallel_compact`,
`_flux2_attn_compact`), after an `index_select` of its input rows. None of
that work is needed: the residual of an inactive row is never read (its
output comes from the output cache, its K/V from the KV cache).

Running the step on the compact sequence `[active text rows, active image
rows]` (gather once after `x_embedder` / `context_embedder`, scatter once
before `sync_with_output_cache`, K/V caches and attention keys full length
as now) removes per block two full-length norm + modulation passes, one or
two residual passes, the zero-fills and the scatters/gathers, 25 blocks x 2
steps. The traffic model puts it at 2.2 GB per step. It is the same math
(LayerNorm, modulation and residual adds are per row), so level 0. The CUDA
graph keys (row-count buckets) and `exec_only` masks (manual masks) keep
working; the compiled per-block graphs change shape, so it is one more
warm-up variant.

### 2.2 Fused q/k/v in the double blocks (bit-identical, small)

`Flux2Attention` keeps separate `to_q / to_k / to_v` (and
`add_q/k/v_proj`); `_flux2_attn_compact` runs 6 int8 GEMMs and 6
activation quantizations per double block. A fused 3072x9216 `Int8Linear`
per stream is 2 GEMMs with the same weight bytes and better tiles at small
M: 40 fewer GEMMs and quantization passes per frame. diffusers' Flux2
declares `_supports_qkv_fusion = False`, so it is a FluxRT-side change.

### 2.3 The int8 epilogue writes int32

`Int8Linear` calls `torch._int_mm` (int32 output) and then
`acc.float() * x_scale * w_scale -> bf16`. For the single block's fused
projection (M = 320, N = 27648) that is 35 MB of int32 written and read
back per call, ~1.8 GB per step over 20 blocks even with the epilogue fused
by Inductor. A W8A8 kernel with the dequant epilogue inside (CUTLASS
`scaled_mm`-style with per-token and per-channel scales writing bf16, as in
vLLM; or a Triton kernel) removes that traffic. Linux only in practice;
worth a microbenchmark at FluxRT's shapes before anything bigger, together
with a check of cuBLAS int8 efficiency on sm_120, which is unmeasured.

### 2.4 Already right, do not redo

Text K/V cached across steps and frames (diffusers 0.42 now ships the same
idea as `apply_text_kv_cache`); timestep modulation and RoPE memoized per
schedule step; per-step masks resolved on the host; CUDA graphs per row
bucket; `context_embedder` skipped when no text row is active; K/V written
in place; the two-step schedule applies the empirical-mu shift the model was
distilled with (`compute_empirical_mu`), and the noise is seeded per frame,
so it is identical from frame to frame (random noise measured 6x more
flicker elsewhere, section 6).

### 2.5 Trivia

`prepare_latents` re-draws the same seeded noise every frame (~0.05 ms);
cacheable per (seed, shape). Cosmetic.

## 3. What diffusers 0.42.1 ships that matters here

- **`Flux2KleinKVPipeline`** and the checkpoint `black-forest-labs/FLUX.2-klein-9b-kv`:
  reference-image tokens go through step 0 at a fixed timestep and their
  K/V are reused on later steps (9B only, non-commercial). FluxRT's spatial
  cache already does this for reference rows; the KV checkpoint is trained
  for reference tokens at a fixed timestep, which is what makes the cache
  exact rather than approximate.
- **`Flux3Transformer2DModel`** *[new]* and the modular pipelines for
  `black-forest-labs/flux-3-image-klein` (+ `-pup`, a Qwen3.5-VL prompt
  upsampler): three token modes (image, condition image, text), 5 per-mode
  blocks each (self-attention inside the mode) + 28 joint single blocks,
  3072 wide, text context = 8 stacked Qwen3-VL hidden layers (20480 wide).
  Condition-image and text tokens are modulated at timestep 0, so their
  per-mode blocks do not depend on the step (computable once per frame),
  and the joint blocks could cache them across steps the way FluxRT caches
  today. The step-distilled "flow map" checkpoints take the timestep each
  step jumps to (`target_timestep`) and run 16 steps without CFG. From the
  config: 15 mode blocks + 28 single blocks x 122.7 M = ~5.3 B parameters,
  about 45 % more per-token work than klein-4B. Whether a flow-map model
  holds up at 2 steps is not stated anywhere reachable.
- **Cache hooks** (`diffusers.hooks`): FirstBlockCache, MagCache,
  TaylorSeer, SeaCache *[new]*, PyramidAttentionBroadcast, TextKVCache;
  Flux2 and Flux3 declare `_repeated_blocks`, so they apply. All of them
  skip work between steps of one image and need >= 8 steps to pay off; at
  2 steps they are moot. FluxRT's cache works between frames, which none
  of them do.
- **Attention backends** in `dispatch_attention_fn`: `sage`,
  `sage_blackwell_hub`, `_flash_3`, `flash_4_hub`, `flex`, `_native_cudnn`
  and more. Switching FluxRT to a kernel-hub backend is a config change
  once a Windows / sm_120 wheel exists (section 4).

## 4. Model-side speedups (research)

### 4.1 Where the time goes, reconciled

Section 1's traffic model and the laptop profile in `perf-notes.md` agree:
on the 4090 the two steps are near the bandwidth floor (int8 weights
7.4 GB per frame, activations ~8 GB with the full-length residual stream),
while the 110 W laptop is also compute-bound: 2.3 TFLOP of INT8 per step
in 10.75 ms is ~215 TOPS effective against a GB203 dense INT8 peak of
~450 TOPS at desktop clocks (less at 110 W), so INT8 there is at about half
its practical ceiling and the kernel is not the problem. What moves the
laptop is a format with twice the peak (FP4) or fewer rows; what moves the
4090 is less traffic (sections 2.1, 2.3) and 4-bit weights.

### 4.2 Precision and quantization

- **Consumer Blackwell peaks** (RTX Blackwell whitepaper via Guru3D): RTX
  5090 INT8 838 dense / FP8 838 / FP4 1676 TOPS; RTX 4090 INT8 661 / FP8
  330. So on sm_120 INT8 no longer beats FP8 and FP4 is the only 2x step.
  The fork measured FP8 e4m3 at 2x bf16 (INT8 3x) and row-wise FP8 slower
  than bf16 **on the 4090**, where FP8's peak is half INT8's. On the laptop
  `torch._scaled_mm` FP8 has the same peak as `_int_mm` and a fused
  dequant epilogue that writes bf16 (section 2.3): worth one afternoon of
  re-measurement there, with the known cost of ~3x INT8's error.
- **W4A4 SVDQuant / NVFP4** is the largest GEMM lever: Nunchaku reports
  NVFP4 ~3.1x bf16 on a 5090 and INT4 ~3.0x W4A16 on a 4090 (FLUX.1 at
  1024^2); Nunchaku-Lite's klein-4B example (RTX PRO 6000, 1024^2, 4
  steps) bf16 3.00 s -> NVFP4 2.27 s, 1.68 s with compile *[snippet]*.
  Estimate for FluxRT: laptop GEMM 24 -> ~13-16 ms per frame (15-20 % of
  the frame), 4090 2-5 ms if the kernels hold up at M ~ 320. Quality is the
  problem: the community klein-9B nunchaku card reports PSNR 17.6 / LPIPS
  0.21 against bf16 and the FLUX.3 klein card LPIPS 0.17 *[snippets]*,
  against INT8's measured LPIPS 0.037 here; NVFP4 FLUX.2-dev is "some
  degradation, usable" in ComfyUI tests. Availability: nunchaku upstream
  (v1.2.1, 25 Jan 2026) supports FLUX.1 / Qwen-Image / Z-Image, **not
  FLUX.2** (PR #926 "FLUX.2 transformer runtime" open since 31 Mar 2026);
  tonera's fork publishes `svdq-fp4/int4_r32` klein-4B/9B weights and
  Windows torch 2.11 cu130 wheels *[snippet]*; Nunchaku-Lite loads W4A4
  checkpoints into stock diffusers with prebuilt Linux kernels (sm_89,
  sm_120a) via the `kernels` hub package; **comfy-kitchen 0.2.41** ships a
  Windows cp312 wheel with SVDQuant W4A4, NVFP4 `scaled_mm` (TN only),
  fused W4A8 and an AWQ W4A16 kernel with a small-M path (1.6x eager at
  M = 256 on a 5090, 0.88x at 512), cubins for sm_75-sm_120, requiring a
  CUDA 13 runtime and driver r580+. BFL also published official
  `FLUX.2-klein-4b-fp8` (4.07 GB) and `-nvfp4` (2.46 GB, Apache-2.0)
  checkpoints; the nvfp4 one is reported noisy in ComfyUI's template with
  a diffusers key mismatch *[snippet, unresolved]*. Integration cost for
  FluxRT: a W4A4 module in the role of `Int8Linear` with SVDQuant's
  rank-32 low-rank branch, captured in the CUDA graphs. Plain NVFP4 via
  `torch._scaled_mm` (torchao `NVFP4InferenceConfig`) passes torchao's
  capability gate on sm_120 but nobody has published a 5090 run; sm_120
  accepts only TN layouts before cuBLAS 13.6 (pytorch #198188).
- **Better INT8 at no speed cost:** `vistralis/FLUX.2-klein-4b-INT8-transformer-quants`
  are torchao W8A8 static / dynamic per-row quants with SmoothQuant
  variants pending; migrating activation outliers into the weights
  (SmoothQuant) lowers W8A8 error with the same `_int_mm` kernel. Cheap to
  test with `perf_ab.py compare --lpips` against the current 0.037.
- **DIY distillation** to 1-2 steps (DMD-style "1.x-Distill", BiFM,
  Qwen-Image-Flash recipes): no public 1-step klein exists; a training
  project, not an optimization.

### 4.3 Step structure

- **Step caches are moot at 2 steps**, confirmed: mlx-teacache skipped 0
  of 8 steps on klein; TeaCache, FirstBlockCache, MagCache, TaylorSeer,
  SeaCache, Chipmunk, RAS, SpecEdit, AViTS and FreezePlan (Oct 2026
  *[new]*, the same spatial-freezing idea as FluxRT's on multi-step
  editors) all need >= 8 steps.
- **Stream batching** (StreamDiffusion ICCV 2025, 1.5x throughput):
  compute step 2 of frame t and step 1 of frame t+1 in one pass. At steady
  state the 4090 is bandwidth-bound, so one weight stream serves two steps'
  rows: up to ~1.3-1.5x generated fps on the 4090 at ~30 % more latency per
  frame; less on the compute-bound laptop. It needs a batch-2 spatial
  cache (each step has its own K/V tensors, so the attention runs with a
  batch of 2 caches) and per-batch modulation. A throughput lever for the
  4090 / batch render, not for the show's latency.
- **Reference-token pruning** (ToPi, >30 % faster on in-context DiTs
  *[snippet]*): with the show's 256x256 reference the reference rows are
  256 of 2208 keys, so the attention / KV-read saving is ~2-3 ms on the
  laptop at a fidelity risk on faces. Low priority.

### 4.4 Attention

Per step the 25 attention calls (queries = active rows, keys = ~2000)
cost 4.5 ms on the laptop with cuDNN, 0.215 ms per call at 320x1952:
~37 TFLOPS effective, the small-M regime. Options: FlashAttention-2
varlen on Windows now has community cp312 wheels for torch 2.11 **cu130**
(wildminder), which would also end the per-query-length cuDNN plan builds
and the row bucketing (a p95 gain more than a mean one, <= 5 %);
SageAttention 2++ has sm_120 kernels in woct0rdho's abi3 wheels (post6+
fixes the black-frame bug) but was measured at 3.85 % error here and
rejected; SageAttention 3 FP4 is faster still and less accurate than SA2;
FlashAttention-4 cannot run on sm_120 (no TMEM / tcgen05) and
FlexAttention would be Triton-dense and slower. The gather / copy share
(5.7 ms per laptop frame) is section 2.1's, not the kernel's.

### 4.5 The conv stages

- **Interpolation on the optical-flow engine.** NVIDIA's FRUC (frame-rate
  up-conversion on the NVOFA hardware of RTX 30/40/50) landed in FFmpeg as
  `fruc_vulkan` on 29 Aug 2026 *[new]*; NVIDIA's app note quotes 4.4 ms
  per 1080p frame on a 4090 *[snippet]* against RIFE-TensorRT's 6.3 ms at
  1152x640 on the laptop, and it runs off the tensor cores. Quality is
  below RIFE on large motion, and the interop (Vulkan images in and out of
  torch) is the cost. Worth a prototype for the production preset, where
  RIFE is 8 ms of real GPU work.
- **RIFE:** no newer model than 4.26 / 4.25.lite (2024); a TensorRT
  >= 10.14 regression report for RIFE 4.25 FP16 (NVIDIA/TensorRT #4721,
  3.5 -> 9.6 ms) — the fork measured a gain with TensorRT 11.3, so verify
  per version.
- **Decoders:** TAEF2 has no 2026 update; BFL's `FLUX.2-small-decoder`
  (28 M parameters, ~1.4x faster than the full decoder) is a middle quality
  option between TAEF2 (LPIPS 0.09-0.2) and the full VAE (33 ms on the
  laptop), i.e. a 4090 option. DC-AE / LiteVAE would need the DiT
  retrained.
- **Upscaler, a hypothesis (not from the research):** the production
  preset spends ~24 ms of GPU per frame on 1152x640 (upscaler UNet 6 ms,
  the larger decode and RIFE); decoding at 576x320 and running a small
  image-space 2x super-resolution net (SPAN / RTMoSR-class, TensorRT
  fp16, ~1-2 ms) on the RGB frames, or leaving the 2x to the kiosk's NIS
  as `OPS-show-machine.md` already suggests, would save ~10 ms at the
  price of hallucinated latent detail. A product decision to test, with
  the perf_ab contact sheet.

### 4.6 Platform

- torch: 2.11 (Mar 2026) -> 2.12 (May, cu128 wheels deprecated,
  `torch.cond` in CUDA graphs) -> 2.14 (Sep, NVFP4 GEMM in Inductor) ->
  2.15 (28 Oct 2026, CUDA 13.2 default) *[new]*. Everything new on
  Blackwell (comfy-kitchen kernels, NVFP4, the FA2 Windows wheels) wants a
  **cu130** torch; the compile and TensorRT caches would be rebuilt.
- triton-windows: woct0rdho's repo was archived on 18 Feb 2026 and the
  project continues as `triton-lang/triton-windows` (PyPI 3.9.0.post30;
  torch 2.11 pairs with 3.6, so `pip install "triton-windows<3.7"` for the
  current stack).
- Keep hardware-accelerated GPU scheduling on (CUDA graphs amortize the
  WDDM launch cost); MCDM on GeForce is unverified and disables the
  display. For A/B runs on the laptop lock Dynamic Boost and the fan
  curve: the enforced limit moves between 92 and 110 W with CPU load.

## 5. The video path on hardware codecs (research)

### 5.1 NVENC and NVDEC are already installed

`pyav-ffmpeg` builds the PyAV wheels with `--enable-nvenc --enable-nvdec`
on Linux and Windows x86_64. Checked on the installed `av 17.1.0` (FFmpeg
8, libavcodec 62): encoders `h264_nvenc`, `hevc_nvenc`, `av1_nvenc`
accepting `yuv420p`, `nv12`, `bgr0`, `bgra`, 4:4:4 and 10-bit; decoders
`h264_cuvid`, `vp8_cuvid`, `av1_cuvid`; options `tune` (ull),
`zerolatency`, `forced-idr`, `rc-lookahead`, `cbr_ld_hq`, `b_ref_mode`,
`nonref_p`, `ldkfs`, `split_encode_mode`. The driver has to be >= 570
(PyAV #1828: the wheels fail on 560 with "minimum required Nvidia driver").
aiortc 1.15 pins `av >= 14, < 18`. Neither ComfyStream nor Daydream Scope
use any of this: both encode with libx264 / libvpx from `.cpu().numpy()`.

### 5.2 Option A: H.264 on NVENC inside aiortc's encode path (1-2 days)

aiortc's `H264Encoder` is replaceable the way `fluxrt/webrtc/vp8_encoder.py`
replaces the VP8 one: open `h264_nvenc` with `preset p1`, `tune ull`,
`rc cbr`, `bf 0`, `rc-lookahead 0`, `zerolatency 1`, `forced-idr 1`
(aiortc's keyframe request sets `pict_type = I`, which on NVENC is a
non-IDR frame unless `forced-idr` is set: the classic cause of PLI loops),
`profile baseline` (what the SDP advertises; Chrome decodes High in
practice), real pts in a 1/90000 time base instead of aiortc's pinned
1/30 (the reason x264 rate control collapsed at FluxRT's variable frame
rate, see `run_webrtc.py`), and feed `bgr0` frames straight from the
output tensor: no I420 conversion at all. Fabi-SPL/remote-desktop does
exactly this on Windows with the stock pip `av` and measured 10.8 ms per
1080p frame including a BGRA upload from system memory; NVIDIA's app note
puts the engine at 805-910 fps for 1080p P1/CBR on Ada, so 2-4 ms per
1152x640 frame is the expectation against 10-14 ms of libvpx, and ~0.2 of
a core per viewer. `aiortc-h264-nvenc` 1.14.0.post9 on PyPI (Jan 2026) is
a whole-fork variant of the same idea (pins `av < 17`; do not install over
aiortc). REMB keeps working: aiortc rebuilds the encoder on a > 10 %
target change as before (NVENC also supports reconfiguration without a
rebuild). The GeForce session limit is 8 concurrent encoders per system
(12 by late-2025 reports *[unverified]*): one per viewer is fine for a
show, and for more viewers one encode can feed every sender (each
`RTCRtpSender`'s track yields the same packets), which also answers the
"encode once for all viewers" item in `perf-notes.md`.

### 5.3 Option B: encode where the tensor lives (3-5 days)

**PyNvVideoCodec 2.2.4** (PyPI, MIT, cp312 wheels for win_amd64 and
manylinux_2_28 *[new]*) encodes H.264 / HEVC / AV1 straight from any
`__cuda_array_interface__` object (a torch CUDA tensor) in ABGR / ARGB /
NV12 (no RGB24: one `torch.cat` with an alpha plane), with `preset P1`,
`tuning_info ultra_low_latency`, `rc cbr`, `bf 0`, `idrperiod`,
`Reconfigure()` for bitrate changes and `FORCEIDR | OUTPUT_SPSPPS` per
frame; it loads the encoder from the driver (no toolkit). Run in the model
or scheduler process, it removes the `.cpu()` download, both 4.4 MB shared-
memory copies and the CPU conversion (~2-4 ms on the critical path), and
the server only packetizes ~10-30 KB of Annex B per frame (~0.05-0.1 core
per viewer). aiortc takes pre-encoded frames: a track may return an
`av.Packet` and `H264Encoder.pack` splits the Annex B access unit into
STAP-A / FU-A RTP packets (`p = av.Packet(annexb); p.pts = n;
p.time_base = Fraction(1, 90000)`). Two things do not reach an external
encoder on that path and need small patches: keyframe requests
(`RTCRtpSender._send_keyframe` only sets a flag the `Frame` branch
consumes) and REMB (`target_bitrate` is written to the stock encoder
object). The shared-memory protocol stays for the GUI and demos; the pack
path is one more consumer. This is the version that makes CUDA IPC
unnecessary.

### 5.4 Option C: the camera decoded on NVDEC (2-3 days)

aiortc's jitter buffer hands complete Annex B access units to a decoder
thread through a module-level `get_decoder` that can be replaced; a
PyNvVideoCodec decoder (`DisplayDecodeLatencyType.ZERO`, `ENDOFPICTURE`
per packet so a frame is emitted without waiting for the next one)
returns NV12 or RGB on the GPU via DLPack, so the camera frame lands on
the GPU already: CPU decode (2-5 ms per frame), the OpenCV resize and the
upload disappear, and the crop / resize becomes a GPU op in the model
process. VP8 is NVDEC-decodable through Blackwell; to make browsers send
H.264 instead, `setCodecPreferences` on the server's receive transceiver
(what ComfyStream does). Constructing the decoder's packet from Python
bytes is how VPF worked but is not in a 2.2 sample *[unverified]*; the
documented in-memory route goes through FFmpeg's parser and needs an AUD
NAL appended to flush each frame.

### 5.5 CUDA IPC and pinned shared memory

On Linux `torch.multiprocessing` shares CUDA tensors across processes
(`cudaIpcGetMemHandle`; `expandable_segments` off). On Windows it does
not: torch's allocator returns `ipcEnabled() = false` under `_WIN32`, the
one attempt (pytorch PR #190978, Win32 NT-handle export) was closed
unmerged on 29 Oct 2026 *[new]*, and the DIY routes (cuda-python VMM
`cuMemExportToShareableHandle` + `DuplicateHandle` + CuPy wrapping, or a
D3D12 shared heap) are one- to two-week projects. Since the show runs on
Windows, option B (encode in the GPU-owning process) is the practical
form of "no CPU copies". `cudaHostRegister` on the existing segments
makes the CPU path asynchronous and ~2x faster per copy, with Windows
pitfalls: WDDM caps pinned memory near 50 % of RAM and the call fails
near it; keep it to one or two frame buffers.

### 5.6 Other transports

GStreamer `webrtcbin` with `nvh264enc` / `nvd3d11h264enc` and the WHIP /
WHEP elements now has Windows PyPI wheels (`gstreamer-python` 1.28.8,
`gstreamer-plugins` 1.28.7 with `gstnvcodec.dll`, `gstwebrtc.dll`,
`gstrswebrtc.dll`), and does NACK / RTX / pacing in C; it is a rewrite of
the signalling and ownership code, zero-copy from torch into
`GstCudaMemory` is unverified from Python, and a 1.26 Windows `nvh264enc`
regression is unresolved. libdatachannel's Python bindings have no Windows
wheels. Pion / mediasoup / str0m sidecars fed Annex B over a socket are
mature but a second runtime. For the kiosk on the same machine the raw
RGBX WebSocket already skips codecs entirely; H.264 over WebSocket +
WebCodecs (2.5 / 4 ms decode-to-present measured elsewhere, 50x less
bandwidth) is a trade of 44 MB/s of copying for an encode + decode, not an
obvious win. WebTransport shows no latency advantage over WebRTC in
published demos. aiortc has no AV1 (a fork commit exists); AV1 would be a
quality lever, not latency.

### 5.7 Two cheap Windows details

aiortc has no RTP pacer: a 3 Mbps keyframe goes out as ~50 back-to-back
packets, and Windows' default 15.6 ms timer makes `asyncio.sleep` coarse
(50 x 3.2 ms sleeps took 756 ms in remote-desktop's measurement).
`winmm.timeBeginPeriod(1)` at server start is a one-liner that also
tightens FluxRT's own output pacing (`send_wait`, the scheduler's holds)
on the show laptop.

## 6. Community practice, 2026

From GitHub issues, wikis and release notes; the Banodoco Discord corpus
and Reddit were not reachable.

- **FLUX.2 klein in the wild.** BFL positions the distilled 4B/9B at a
  4-step schedule "for production and real-time"; guides agree more steps
  overcook it, and nobody published a 1- vs 2-step comparison. The only
  hard per-step 4090 number: klein 4B at ~10.5 it/s in ComfyUI vs 6.7 in
  stable-diffusion.cpp (leejet, sd.cpp #1215, Jan 2026; resolution of that
  row not restated). NVIDIA / BFL claim NVFP4 2.5-2.7x and FP8 1.6x over
  bf16 at 1024x1024 on RTX 5080/5090 *[snippet]*; a community NVFP4
  nunchaku-lite build of klein-9B reports 1.8x over bf16 and its INT4 build
  slower than bf16 *[snippet]*. **Nunchaku has no official FLUX.2 / klein
  support** (issue #883 closed inactive 11 Mar 2026; PR #926 "FLUX.2
  transformer runtime" open since 31 Mar 2026).
- **Upstream FluxRT quantization data** (issue #12, RTX 3090, Windows,
  448x256, 2 steps): bf16 3.23 fps, quanto int8 3.02, bitsandbytes NF4 4.59
  (PSNR ~14 dB against bf16, "visibly farther"); the maintainer's position
  (#11) that "quantized models help VRAM, not latency" is about quanto's
  weight-only dequant, not the fork's W8A8 `_int_mm` path (3x a bf16 GEMM).
- **InvokeAI PR #9556** *[new]* (klein img2img posterization): noise was
  divided by the VAE's `bn_std` (1.77), the mu shift was skipped, and a
  clipped schedule collapsed to 1-2 steps; the fix keeps noise N(0,1),
  applies the shift and re-spaces the remaining sigmas. FluxRT does not run
  img2img strength and already shifts, so nothing to fix; it is the
  reference for any experiment with the 2-step schedule's middle sigma.
- **Flicker, measured** (aptrn/ScreenDiffusion #32, 4090, Sep 2026
  *[new]*): fixed seed 1.49, per-track 1.74, random 8.72 (6x worse); an
  output EMA of 0.25 cut flicker 14 % and responsiveness 13 %. FluxRT's own
  floor: GPU nondeterminism amplified by the change detector (perf-notes,
  LPIPS 0.018 between two identical runs). Literature for recomputed
  regions: flow-warped noise (Go-with-the-Flow), Warped Diffusion,
  StreamV2V's feature bank.
- **Windows + Blackwell kernels.** `triton-windows` 3.6.0.post25 needs
  torch >= 2.10 and the repo was **archived on 18 Feb 2026** (pin
  `<3.7`; torch 2.12+ unclear). SageAttention: woct0rdho's abi3 wheels
  "torch2.10.0andhigher" load on 2.11; post6 (17 Jul 2026 *[new]*) fixes an
  out-of-bounds bug that produced black / noisy output; sm_120 gets
  SageAttention2++ kernels (CUDA >= 12.8); CUDA 12 and 13 wheels are not
  interchangeable; compile blocks, not the whole graph (full-graph compile
  with Sage fails or goes black). mobcat40 measured Sage 2.2 on a 5090 at
  30-35 % faster than SDPA for Qwen-Edit (torch 2.11, cu128 and cu130).
  sdbds/SageAttention-for-windows ships cu130 wheels for torch 2.10-2.14
  (latest 24 Sep 2026 *[new]*) and a SageAttention3-Blackwell build.
  flash-attn: no upstream sm_120 Windows wheels; community cp312
  torch-2.11/cu130 builds exist *[snippet]*. NVFP4 in ComfyUI needs a cu130
  torch. HAGS off costs 3-5 %.
- **Expectations the community reports** (useful for the show):
  upstream users see 3-7 model fps on a 4090 under Windows, FluxRT-TD
  (TouchDesigner wrapper) 4.7 model fps on a desktop 5090 with a webcam
  scene; 16 GB cards are not viable; "Windows gets 65-75 % of Linux". The
  fork's 12-18 fps on a 110 W 5090 Laptop is above anything reported on
  desktop 5090s under Windows.

## 7. Alternatives: the real-time video-to-video landscape

Upstream and ecosystem first, then the models.

**Upstream tensorforger/FluxRT** (read 2026-11-05): 471 stars, 51 commits,
Unlicense, HEAD `32206e8` "flow upscaler and taef2" of 13 June 2026, no
commit since; maintainer's last activity 4 June (issue #23). Open issues
are all performance reports: #21 (Dynamo `recompile_limit` hit, 0.5 fps
until warm; the fork fixed this class with per-block compile and the
warm-up sweep), #23 (5090 at 2 fps), #26 (16 GB card, 1 fps), #12 / #11
(quantization, above), #3 (benchmarks: 4090 Windows 145-339 ms per
generated frame against the fork's 32.5 ms on Linux). PRs: the fork's #25
(WebRTC, 59 commits) and #28 (input ownership) were closed by their author
after no maintainer response; the only upstream-side artefact is the open
Intel XPU PR #22. `git merge-base` of the fork's `main` and upstream HEAD
is upstream HEAD: the fork is a superset. Of 64 forks none has comparable
performance work (gschian0: show deployment scripts; KurtKomell: batch
scripts; keishidev: an MPS attempt). Third-party: MatthiasHassel/FluxRT-TD
(TouchDesigner, Windows, MIT) and Deep-Live-Cam 2.7's "FLUX Live" mode,
"powered by FluxRT", gated to 5090-class GPUs.

**Daydream Scope** (daydreamlive/scope, CC BY-NC-SA 4.0 alpha): v0.2.5 of
15 May 2026 is the last release and the commit log shows nothing after it
*[unverified whether work moved]*. Pipelines: LongLive, StreamDiffusionV2,
Krea Realtime, RewardForcing, MemFlow, LTX-2 (batch clips, "no
frame-by-frame streaming"); plugins include hthillman/scope-flux-klein
(feedback-loop img2img, 2-step default). The byMARCO "FluxRT" plugin that
the README links could not be found in source form *[unverified]*.
Video-to-video is chunked with SDEdit-style `noise_scale`; reference
images only via VACE, which Daydream's own paper calls "significantly
degraded" under causal attention. Best community numbers come from a
Blackwell-patched fork on an RTX 5080 Laptop (jordancoult/ai-livestream,
Nov 2026 *[new]*): StreamDiffusionV2 38.7 fps model-only at 576x320,
LongLive 1-step 21 fps at 640x352, stock Scope 3.6 fps before patches.

| Model | Type | Size | Best number | Conditioning (camera / prompt / reference) | License | Windows | State |
|---|---|---|---|---|---|---|---|
| **FluxRT, this fork** | per-frame instruct edit + spatial cache + RIFE | klein-4B, 3.7 B | 30.8 gen fps 576x320 on 4090 Linux (61 shown); 12-18 on 5090 Laptop 110 W | yes / yes / yes | Unlicense; model Apache-2.0 | yes | production |
| FLUX.2 klein 9B / 9B-KV | per-frame edit, 4-step distilled | 9 B + 8 B TE | ~2 s per image on 5090 *[snippet]* | - / yes / yes (KV cache) | non-commercial | yes | stable |
| FLUX.3 Image Klein *[new]* | per-frame edit, 16-step distilled (base 50) | 5.5 B + 4 B Qwen3-VL | ~4 s per MP on 5090 *[snippet]*; ~20 GB bf16 | - / yes / up to 10 refs + layout | FLUX Kommunity (free under $5 M revenue) *[snippet]* | yes (diffusers) | 3 weeks old |
| Qwen-Image 2.1 + Lightning *[new]* | per-frame edit, 4-step LoRA | 7 B | none published | - / yes / up to 10 refs | research license | yes | new |
| StreamDiffusion (SD-Turbo/SDXL, TensorRT) | per-frame img2img | 1-3 B | 93.9 fps 1-step 512x512 on 4090 | yes / yes / IP-Adapter only, no instruct edit | Apache-2.0 | plausible | mature |
| **StreamDiffusionV2** | autoregressive video (Wan 1.3B / 14B) | 1.3 B | 4x4090 ~16 fps at 480p; 38.7 fps at 576x320 on a 5080 Laptop (community) | chunked video + noise_scale / yes / VACE (poor) | Apache-2.0 | no | MLSys'26 best paper; Blackwell since May 2026 |
| LongLive 1.3B / 2.0 5B *[new]* | autoregressive video | 1.3 B / 5 B | 45.7 fps 2-step NVFP4 on GB200; 13-15 fps 1024x576 on 5080 Laptop | text, I2V; no v2v | Apache-2.0 | no | active |
| RewardForcing / MemFlow / Self-Forcing | autoregressive video | 1.3 B | 17-23 fps (H100 / laptop 5080 in Scope) | v2v in Scope + VACE | Apache-2.0 | no | research in Scope |
| Krea Realtime 14B | autoregressive video | 14 B | 11 fps on B200, 40 GB+ | yes / yes / VACE (~55 GB) | CC BY-NC-SA | no | 4 commits |
| **SANA-Streaming** (NVIDIA) *[new]* | streaming video *editing* | 2 B + LTX-2 causal VAE + Gemma-2-2b | **24 fps at 1280x704 on one RTX 5090**, 4 steps, 5.6 GB | yes / instruction / **no reference** | repo Apache-2.0; weights *[unverified]* | no | 5 months |
| **JoyAI-Video-Edit** (JD) *[new]* | streaming video editing | 16 B + 7 B encoder | **24 fps at 840x480 on one 5090 (32 GB)**; ~0.46 s GPU, ~1 s glass-to-glass | yes / instruction / **reference image** (garment, subject) | Apache-2.0 | no | HF Space with webcam |
| LiveEdit (ECCV'26) *[new]* | streaming video editing | 1.3 B | 12.66 fps, 4 steps (GPU unstated) | yes / instruction / - | Apache-2.0 | no | research |
| LTX-2 / 2.3 / 2.5 *[new]* | batch clips (8-step distilled) | 22 B | 54 s per 121 frames at 960x544 on 5090 *[snippet]* | i2v / t2v / guide video; no live v2v | LTX open license | yes | not real-time |
| TurboDiffusion | offline clip acceleration | Wan 1.3 B | 5 s of 480p in 1.9 s on 5090 | t2v / i2v | Apache-2.0 | - | active |

What it means:

- **Same role as klein-4B** (drop into FluxRT's cache / RIFE / WebRTC
  stack): klein 9B (>= 4x the cost, non-commercial) and FLUX.3 Image Klein
  (same FLUX.2 autoencoder, so latents, TAEF2 and the upscaler carry over;
  5.5 B at 16 steps is two orders of magnitude off 30 ms per frame; the
  spatial cache would be re-implemented on `transformer_flux3.py`, which
  its per-mode design makes easier). Nothing beats klein-4B at 2 steps
  today.
- **Different paradigm:** the autoregressive models replace the whole
  diffusion + cache + RIFE block and take temporal coherence from the model
  (no change mask). The two to test are SANA-Streaming (4x FluxRT's pixel
  rate on a 5090, but no reference image) and JoyAI-Video-Edit (reference-
  guided live editing, i.e. the try-on use case, at ~4x FluxRT's latency,
  32 GB and Linux only). LiveEdit's "AR mask cache" is FluxRT's idea inside
  an autoregressive editor.
- **Scope is not a shortcut:** dormant, non-commercial, weak reference
  path; if Scope reach matters, publishing FluxRT's own plugin (a
  `Pipeline` with `Requirements(input_size)` over THWC tensors) is the
  safer route.

## 8. Ranked recommendations

Ordered by expected gain per unit of effort and risk, for the show
machine first. "Level" is `perf-notes.md`'s output-change scale (0 =
bit-identical, 1 = rounding, 2 = measurable, 3 = different look).

| # | Change | Expected | Level | Effort | Platform | Evidence | First experiment |
|---|---|---|---|---|---|---|---|
| 1 | H.264 on NVENC through PyAV inside aiortc (5.2), `bgr0` input | -8 to -12 ms encode per viewer per frame; server CPU ~1.4 -> ~0.4 core at 1 viewer | 1 (codec change) | 1-2 days | both; driver >= 570 | verified wheel + remote-desktop's working patch | `perf_webrtc.py loopback`, then chrome://webrtc-internals keyframes / bitrate |
| 2 | Compact residual stream in the transformer (2.1) | -3 to -5 ms per frame on the 4090, more on the laptop | 0 | 2-3 days + GPU time | both | traffic model; code reading | `perf_ab.py` A/B, 300/300 identical frames expected |
| 3 | `winmm.timeBeginPeriod(1)` on Windows (5.7) | fewer late frames / keyframe bursts; p95 | 0 | 1 hour | Windows | remote-desktop measurement | live `/healthz` + kiosk `lag` |
| 4 | Fused q/k/v in the double blocks (2.2) | -0.5 to -1 ms per frame | 0 | 1 day | both | code reading | perf_ab |
| 5 | Re-measure FP8 `_scaled_mm` for the block linears on the laptop (4.2) | unknown; FP8 = INT8 peak on sm_120 and no int32 epilogue | 2 (3x INT8's error) | half a day | 5090 | whitepaper peaks | microbenchmark at M = 320, then perf_ab LPIPS |
| 6 | SmoothQuant scales for the existing W8A8 path (4.2) | quality only (LPIPS 0.037 -> ?), same speed | improves | 1-2 days | both | vistralis quants exist | perf_ab `--lpips` |
| 7 | NVENC in the model process + packet fan-out + NVDEC inbound (5.3, 5.4) | -2 to -4 ms critical path; server ~0.1 core per viewer; N viewers at one encode | 1 | 1-2 weeks | both (PyNvVideoCodec wheels) | API verified from the wheel; keyframe / REMB patches needed | loopback with `--viewers 3/5` |
| 8 | FRUC (hardware optical flow) instead of RIFE (4.5) | -3 to -5 ms on the production preset, off the tensor cores | 2-3 | 1-2 weeks | both (Vulkan) | NVIDIA figure *[snippet]*; FFmpeg filter *[new]* | perf_ab flicker + contact sheet on the webcam clip |
| 9 | 2x in image space (SR net or the kiosk's NIS) instead of the latent upscaler (4.5) | ~-10 ms on the production preset | 3 | days | both | hypothesis | side by side with the contact sheet; a product decision |
| 10 | W4A4 SVDQuant / NVFP4 block linears (4.2) | laptop GEMM 24 -> 13-16 ms (-15-20 % frame); 4090 -2 to -5 ms | 2-3 (reported LPIPS ~0.2) | 2-4 weeks; cu130 torch | 5090 (NVFP4), 4090 (INT4) | nunchaku numbers for FLUX.1; klein support only in forks | one block in isolation first, at M = 320 |
| 11 | Stream batching of two frames' steps (4.3) | +30-50 % generated fps on the 4090 at +30 % latency | 0 | 2-3 weeks | 4090 / batch render | StreamDiffusion 1.5x | batch-render throughput |
| 12 | cu130 torch + triton-windows 3.9 migration (4.6) | enabler for 5, 7, 10 and FA2 on Windows | 0-1 | days + cache rebuilds | Windows | release notes | perf_ab before / after |
| 13 | Benchmark the streaming editors next to FluxRT (7) | a different product trade-off, not a speedup | - | days | Linux 4090 / 5090 | published fps | SANA-Streaming on the 4090, JoyAI-Video-Edit on a 5090, perf_ab's LPIPS / flicker harness |

Not recommended now, with the reason: step caches of any kind (no steps
to skip); SageAttention 2 / 3 (accuracy, measured and reported);
TensorRT-RTX for the DiT (static shapes, FP8 quality, high effort; its
klein demo rejects image conditioning); FLUX.2 klein 9B / 9B-KV
(non-commercial, >= 4x the cost); FLUX.3 Image Klein as a drop-in (16
steps, 5.5 B; revisit if a few-step distillation ships); CUDA IPC on
Windows (not in torch; option B gives the same result); Daydream Scope as
a host (dormant, non-commercial, weak reference path); GStreamer /
sidecar transports (a rewrite for a bottleneck the loopback does not
show at 1152x640).

## Sources

Code read locally: FluxRT `src/fluxrt/stream_processor/{pipeline,transformer_flux2,int8_linear,update_controller,interpolation_model}.py`,
`src/fluxrt/flow_upscaler/*`; diffusers 0.42.1 (`pipelines/flux2/*`,
`models/transformers/transformer_flux2.py`, `transformer_flux3.py`,
`modular_pipelines/flux3/*`, `hooks/*`, `models/attention_dispatch.py`);
aiortc 1.15.0 (`rtcrtpsender.py`, `rtcrtpreceiver.py`, `codecs/h264.py`,
`codecs/vpx.py`, `jitterbuffer.py`); the installed PyAV 17.1.0 wheel; the
PyNvVideoCodec 2.2.4, torchcodec 0.17.0, GStreamer 1.28 and comfy-kitchen
0.2.41 wheels (pip-downloaded); clones of tensorforger/FluxRT,
livepeer/comfystream, daydreamlive/scope, chenfengxu714/StreamDiffusionV2,
Fabi-SPL/remote-desktop, PyAV-Org/pyav-ffmpeg, pytorch/ao.

FluxRT and forks: github.com/tensorforger/FluxRT (issues 3, 11, 12, 21,
23, 26; PRs 22, 25, 27, 28; benchmark.md), github.com/204-ai/FluxRT (PRs
19, 26, 28, 34), MatthiasHassel/FluxRT-TD, KurtKomell/FluxRTPerformanie,
UltraCombos/FluxRT, hacksider/Deep-Live-Cam discussions/1837,
huggingface.co/black-forest-labs/FLUX.2-klein-4B/discussions/14 *[snippet]*.

Models and quantization: github.com/black-forest-labs/flux2 (and
docs/flux2_klein_kv_cache.md); huggingface.co/black-forest-labs
FLUX.2-klein-4B, -4b-fp8, -4b-nvfp4, -9b-kv, FLUX.2-small-decoder
*[snippets]*; huggingface.co/blog/flux-3 (via the blog repo on GitHub);
venturebeat.com on FLUX.3 Image Klein; github.com/huggingface/diffusers
(releases; docs/source/en/api/pipelines/flux3.md); aydin99/FLUX.2-klein-4B-int8,
vistralis/FLUX.2-klein-4b-INT8-transformer-quants, unsloth/FLUX.2-klein-4B-GGUF,
tonera/FLUX.2-klein-{4B,9B}-Nunchaku, tonera/flux-3-image-klein-nunchaku
*[snippets]*; github.com/nunchaku-ai/nunchaku (PR 926, releases v1.2.1,
issue 883); huggingface.co/blog/nunchaku-diffusers; github.com/rootonchair/nunchaku-lite;
github.com/Comfy-Org/comfy-kitchen; FurkanGozukara/Stable-Diffusion wiki
(BF16 vs GGUF / FP8 / NVFP4); blogs.nvidia.com rtx-ai-garage; Guru3D on the
RTX Blackwell whitepaper; github.com/lna-lab/blackwell-geforce-nvfp4-gemm;
NVIDIA/cutlass #3096; deepseek-ai/DeepGEMM #236; pytorch/pytorch #198188
and releases; pytorch.org blog "faster diffusion on Blackwell"; NVIDIA
TensorRT-RTX (PR 44, demo/flux2), pytorch/TensorRT tensorrt_rtx docs,
NVIDIA/Model-Optimizer examples/diffusers; leejet/stable-diffusion.cpp #1215;
invoke-ai/InvokeAI PRs 9556, 9955; kijai/ComfyUI-KJNodes #547;
IonDen/mlx-teacache #80; sandyresearch/chipmunk; arXiv 2610.34177
(FreezePlan), 2602.01609 (ToPi), 2604.04018, 2606.03746, 2605.02152,
2608.17995, 2606.13496, 2608.09231, 2512.16093 *[snippets]*; ICCV 2025
StreamDiffusion paper.

Attention, platform: github.com/woct0rdho/SageAttention (releases),
thu-ml/SageAttention, mengqin/SageAttention, mobcat40/sageattention-blackwell,
sdbds/SageAttention-for-windows, wildminder/AI-windows-whl,
Rogala/AI_Attention, the solatticus gist on FA4 / sm_120,
huggingface.co/nhathoangfoto/SageAttention-3-Blackwell-FP4 *[snippet]*,
Comfy-Org/ComfyUI discussions 11583 and 6643, issue 11739, PRs 16438 and
11890; triton-lang/triton-windows and woct0rdho/triton-windows (releases,
#158); hzwer/Practical-RIFE; NVIDIA/TensorRT #4721; NVIDIA optical-flow SDK
NVOFA app note *[snippet]*; Guru3D on FFmpeg `fruc_vulkan`; ASUS ROG /
Tom's Hardware / Notebookcheck on RTX 5090 Laptop power.

Transport: pypi.org/project/pynvvideocodec and its NVIDIA docs *[snippets]*;
NVIDIA NVENC / NVDEC application notes *[snippets]*; PyAV-Org/pyav-ffmpeg,
PyAV discussions 1828, issue 596; aiortc discussions 769 / 965, issues 323,
588, changelog; pypi aiortc-h264-nvenc; Fabi-SPL/remote-desktop;
anganyAI/aiortc PR 1; meta-pytorch/torchcodec; pytorch Windows notes,
PR 190978, c10/cuda/CUDACachingAllocator.cpp; NVIDIA forum threads on
cudaHostRegister / pinned memory on Windows (228235, 77439, 279429) and
VMM (148875); gstreamer.freedesktop.org releases 1.26 / 1.28 and the
rswebrtc / webrtchttp docs, discourse thread 4391; pypi gstreamer-bundle,
pylibdatachannel, libdatachannel-py, aiolibdatachannel; pion/webrtc #1652;
mediasoup discourse 5891; str0m; webrtc.rs rtc 0.3.0; GoogleChrome
webtransport sample, MDN WebTransport, webrtcHacks; dev.to (H.264 +
WebCodecs latency), vidgear PR 468, punktfunk/client-web #53;
livepeer/comfystream and docs.livepeer.org *[snippet]*; daydreamlive/scope
(docs/server.md, docs/vace.md, docs/plugins.md, docs/api/parameters.md,
issue 917, releases); jordancoult/ai-livestream; hthillman/scope-flux-klein.

Alternatives: chenfengxu714/StreamDiffusionV2 and arXiv 2511.07399;
krea-ai/realtime-video; NVlabs/LongLive; NVlabs/Sana
(docs/sana_streaming.md) and arXiv 2605.30409; jd-opensource/JoyAI-Video-Edit
and arXiv 2608.03974; cp-cp/LiveEdit and arXiv 2606.26740;
guandeh17/Self-Forcing; thu-ml/TurboDiffusion; daydreamlive/StreamDiffusion;
olegchomp/TouchDiffusion; aptrn/ScreenDiffusion #32; dotsimulate
StreamDiffusionTD notes; comfyui-wiki on Qwen-Image 2.1 Lightning and
LTX-2.5 *[snippets]*; arXiv 2602.14381 (Daydream), 2606.05981, 2601.16933,
2506.08009 *[snippets]*; the CVPR 2026 StreamDiT page *[snippet]*;
yepicaiaaron/awesome-realtime-video-generation.
