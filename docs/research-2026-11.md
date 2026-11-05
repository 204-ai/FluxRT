# Research: where FluxRT can still get faster, and what else exists (November 2026)

> **Draft in progress (2026-11-05):** sections 4 (model-side speedups), 5 (video path) and 8 (ranked recommendations) are still being researched and will land in a follow-up commit.

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
5. Model-side and transport-side candidates with numbers: sections 4 and 5.

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

(to be merged)

## 5. The video path on hardware codecs (research)

(to be merged)

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

(to be written)

## Sources

(to be merged)
