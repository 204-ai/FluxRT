# Performance notes

What was measured, changed, tried and dropped while speeding up the live
pipeline (September–October 2026), on a Linux RTX 4090 box and on a Windows
laptop with an RTX 5090 Laptop GPU. Read this before starting another round of
performance work: several plausible ideas are already ruled out, with numbers.

Scope: FLUX.2-klein-4B, 2 steps, 576×320, spatial cache on, RIFE ×2
(`--interp 1`). "fps generated" is frames out of the diffusion pipeline; the
stream shows twice that after interpolation.

## How to measure

`scripts/perf_ab.py` drives the live code path (`step_live()`) one fixed input
frame at a time, so every configuration sees the same sequence and the spatial
cache evolves the same way.

```bash
python scripts/perf_ab.py run --repo . --config configs/config_with_reference.json \
  --set interpolation_exp=1 --clip webcam.mov --frames 300 --warmup 40 --out runs/mine
python scripts/perf_ab.py compare runs/main runs/mine --noise runs/main2 --lpips --out runs/report
```

- `--set KEY=VALUE` overrides any config key (same syntax as `run_webrtc.py`).
- `--profile K` profiles K extra frames: per-stage ms, kernel groups, GPU busy %.
- `--live-warmup` runs the server's boot warm-up (`warm_up()`) before the
  clip. Without it the harness skips what the live server does at boot, and
  the timed frames also pay the first use of every new active-row count.
- `compare` reports speed, LPIPS, structure PSNR at quarter resolution,
  frame-to-frame flicker, and writes a contact sheet and a side-by-side video.
- Stop the model server first. Nothing else may share the GPU.

**Noise floor.** The same code run twice on the same clip gives 0 of 300
identical frames (LPIPS 0.0183, structure PSNR 32.8 dB): GPU rounding
differences are amplified by the change detector and the two diffusion steps.
"Identical output" is therefore not a usable bar across runs. A change counts
as quality-neutral when it lands inside that spread; pass the second baseline
run as `--noise`. Timing noise between two baseline runs was 0.2 ms.

**Levels of "unaltered output".** Every change gets one, with its number:

| Level | Meaning | Check | Policy |
|---|---|---|---|
| 0 bit-identical | Same numbers, every element | Old and new path from the same state, compared for equality | On by default |
| 1 rounding | Same math in another order or layout, ~1% relative difference per stage | Full-pipeline LPIPS inside the spread of two runs of the old code | On by default, toggle kept |
| 2 measurable | LPIPS 0.03–0.09 | LPIPS, structure PSNR, flicker, frames | Opt-in |
| 3 different look | LPIPS ~0.2 | Same, by eye | Opt-in |

Most of the "rounding" difference between stage variants is cuDNN picking
another convolution algorithm (`cudnn.benchmark` decides by timing, per
process), which also happens between two runs of the same code.

Rules that held up:

- Same clip, same warm-up length, same flags. A different `--warmup` shifts the
  measured window to frames with different motion, so runs are only comparable
  with the same value (the laptop runs below use 120, the 4090 runs 40).
- One run is not a result when the machine state changed. The first run after
  a reboot was 5 ms slower than its repeat (cold disk and compile cache), and
  one of two identical runs once came out 18 ms slower for no visible reason.
- A/B inside one process only works when every pass plays the same frames.
  Passes over different parts of the clip made a 3 ms gain look like 7.
- Report mean, p50 and p95. A mean far above the median means stalls, not
  slow frames (see "cuDNN plans" below).

## Switches

Everything with a visible effect is a config key, off by default.

| Key | Default | Effect | Output |
|---|---|---|---|
| `int8_linear` | `false` | `true`: W8A8 INT8 for all block linears (`torch._int_mm`). `"inputs"`: input-side projections only | LPIPS 0.037 / 0.031 |
| `vae_decoder` | `"full"` | `"taef2"`: tiny decoder, full encoder | LPIPS 0.086, softer detail |
| `enable_tiny_vae` (`--tiny-vae`) | `false` | TAEF2 on both sides | LPIPS 0.20, different look |
| `mask_dilation` | `2` | `1`: fewer recomputed tokens | LPIPS 0.049, less flicker, staler edges |
| `mask_threshold` | `0.1` | change-detector threshold | visible |
| `rife_cudagraphs` | `false` | RIFE replayed as a CUDA graph | none |
| `enable_flow_upscaler` (`--flow-upscaler`) | `false` | 2× latent upscale, 1152×640 output | more detail, much slower on small GPUs |
| `prompt_travel_refresh` | `"stride"` | `"rolling"` spreads the full recompute | visible during travel |

Quality-neutral keys (on by default, kept as toggles for A/B):

| Key | Default | What it does |
|---|---|---|
| `compile_regional` | `true` | Compile per transformer block instead of one whole graph |
| `compile_vae` | `true` | Compile the VAE encoder / decoder modules actually used |
| `cudnn_benchmark` | `true` | cuDNN picks the fastest conv kernels |
| `warmup` | `true` | Run the known shape variants once at startup |
| `attention_backend` | `"auto"` | cuDNN attention first where flash attention is missing (Windows) |
| `active_row_bucket` | `64` | Round the active-row count up to a multiple (1 = exact) |
| `warmup_sweep` | `true` | Boot warm-up runs every bucketed active-row count once |
| `transformer_cudagraphs` | `true` | Transformer step replayed from recorded CUDA graphs (level 0) |
| `compile_upscaler` | `true` | Compile the flow upscaler's UNet (level 1) |
| `tiny_vae_channels_last` | `true` | TAEF2 in NHWC memory layout (level 1) |
| `stage_cudagraphs` | `true` | VAE encoder / decoder and upscaler UNet replayed from CUDA graphs (level 0) |

Opt-in, level 1, needs extra packages:

| Key | Default | What it does |
|---|---|---|
| `conv_backend` | `"torch"` | `"tensorrt"`: TAEF2, upscaler UNet and RIFE as TensorRT fp16 engines. Needs `tensorrt-cu12` and `onnx` |
| `gpu_wait` | `"spin"` | `"sleep"`: the frame loop sleeps in 0.5 ms steps while the GPU finishes a frame instead of spinning a CPU core in the output download. Same output. Not benchmarked yet |
| `host_masks` | `true` | Resolve per-step masks on the host, one download per frame |
| `area_downscale` | `true` | Area filter for input downscaling (the old call silently used bilinear) |

## Results: Linux, RTX 4090

300-frame webcam clip, old code (`main` before PR #26) as reference.

| Preset | Flags | ms / frame | fps generated | Speed | LPIPS | VRAM |
|---|---|---:|---:|---:|---:|---:|
| Old code | | 141.1 | 7.1 | 1.00× | — | 17.9 GB |
| Default | | 80.3 | 12.5 | 1.76× | 0.016 (noise) | 17.6 GB |
| INT8, input layers | `int8_linear=inputs` | 68.2 | 14.7 | 2.07× | 0.031 | 15.1 GB |
| INT8 | `int8_linear=true` | 65.3 | 15.3 | 2.16× | 0.037 | 14.2 GB |
| Tighter mask | `mask_dilation=1` | 75.3 | 13.3 | 1.87× | 0.049 | 17.6 GB |
| Fast, full VAE | `int8_linear=true mask_dilation=1` | 55.4 | 18.1 | 2.55× | 0.074 | 14.1 GB |
| Tiny decoder | `vae_decoder=taef2` | 63.8 | 15.7 | 2.21× | 0.086 | 17.3 GB |
| Tiny decoder + INT8 | `vae_decoder=taef2 int8_linear=true` | 44.8 | 22.3 | 3.15× | 0.091 | 14.1 GB |
| Tiny VAE | `--tiny-vae` | 59.3 | 16.9 | 2.38× | 0.200 | 17.4 GB |
| Tiny VAE + INT8 | `--tiny-vae int8_linear=true` | 38.4 | 26.0 | 3.68× | 0.202 | 13.9 GB |
| All-in | `--tiny-vae int8_linear=true mask_dilation=1 rife_cudagraphs=true` | 32.5 | 30.8 | 4.35× | 0.210 | 14.0 GB |

LPIPS around 0.1 is where differences become noticeable side by side.

Where the default path's 60.8 ms came from (cumulative, same clip):

| Change | ms saved |
|---|---:|
| Resolve changed tokens once per step. The spatial cache asked the GPU ~140 times per step which tokens changed (a sync each time) and the compiled model broke into ~50 pieces | 36.6 |
| Actually compile the VAE. The old compile call wrapped a method the pipeline never uses | 8.0 |
| Single blocks work on changed tokens only (projections, norms, in-place KV write) | 12.4 |
| Same for the double blocks | 1.5 |
| `cudnn.benchmark` | 2.3 |

GPU busy time went from 69% to 91%: the old code was mostly waiting.

## Live-use lessons

- **Whole-graph compile stalls a live session.** Adding a reference image,
  a new active-row count or a prompt-travel state triggered a 30–60 s
  recompile mid-session (output frozen). Fixes: per-block compile
  (`compile_regional`), the row count marked dynamic, `warm_up()` covering the
  known variants at startup, and a persistent Inductor cache
  (`~/.cache/fluxrt/torchinductor`). A new variant now costs about 1 s.
  Per-block versus whole-graph compile differs by at most 3 ms per frame.
- **Prompt travel** recomputes all 512 text tokens every frame and everything
  every `prompt_travel_full_execute_every` (3) frames. Dips during travel are
  expected.
- **Boot** takes about 60 s of warm-up before the first frame, less with a
  filled compile cache.
- The kernel choice (attention backend) is baked into cached compile
  artifacts. A different backend needs a different cache directory; the code
  appends `-cudnn-attn` for the cuDNN-first order.

## Windows laptop (MSI Stealth A16 AI+, RTX 5090 Laptop 24 GB)

### Hardware limits

- GPU power is 95 W plus 15 W Dynamic Boost, **110 W maximum** for this
  chassis (driver values and MSI's spec agree). The 150–175 W figures belong to
  thick chassis with the same chip. The enforced limit moves between about 92
  and 110 W with CPU load.
- A raw bf16 GEMM at the transformer's shapes takes 1.52 ms here against
  0.84 ms on the 4090: the silicon at this power is about 1.8× slower.
- The MSI profile must be on **Extreme**. On the default profile the GPU
  enforced about 88 W and clocked around 1800 MHz (default preset 211.6 ms
  instead of 183.6 ms).

### Windows-specific findings

- **No flash attention in PyTorch's Windows wheels** ("No available kernel").
  Scaled-dot-product attention falls to the memory-efficient kernel
  (0.50 ms at 704×1952) while cuDNN's takes 0.215 ms. `attention_backend:
  "auto"` puts cuDNN first in the priority order. Disabling flash and
  mem-efficient instead selects the math kernel (3.26 ms), so it has to be a
  priority order, not a disable.
- **The first transformer step at a never-seen active-row count costs
  ~380 ms.** cuDNN plans its attention kernel per query length. With the
  count changing every frame this caused random half-second stalls (mean
  229 ms against a median of 113 ms). Two parts to the fix:
  1. Round the count up to a multiple of 64 by repeating the last index (a
     repeated row computes the same value and scatters to the same place).
     About 25 counts remain.
  2. Run every count once at boot through the real code path
     (`warmup_sweep`, part of `warm_up()`): one frame per count, alone and
     with the text tokens active, again with a reference image if configured.
- **Synthetic priming does not work.** An earlier version built the attention
  plans (66, 22 s) and ran the INT8 matmul shapes (3500, 1.7 s) from zero
  tensors at startup. The first live step at each count still cost ~380 ms:
  the plan also depends on the stride layout of the real (permuted) tensors.
  Removed. Boot went from 115 s to 100 s and the stalled live frames from 6
  to 0 (clip played from the first live frame after boot).
- **A smaller bucket is not worth it.** `active_row_bucket=32` halves the
  padding but measured the same mean (60.8 ms) and doubled the warm-up
  (205 s against 93 s) and the cached memory.
- **SageAttention** was 20% faster than cuDNN (0.173 ms) with 3.85% error, and
  one variant crashed ("no kernel image"). No matching `flash-attn` wheel for
  this torch / GPU. cuDNN stays.
- **Kernel launches are expensive under WDDM.** A sampling profile (py-spy,
  native) of the all-in preset: `cuLaunchKernel` about 28% of the main thread
  (13% inside `NtGdiDdDDISubmitCommandToHwQueue`), `torch._int_mm` call
  overhead 12%, cuDNN attention 6%, Python 20%. The pipeline issues about
  1,000 launches per frame.
- **One CPU core is saturated, the other 23 idle.** One Python thread issues
  all GPU work in order (0.99 cores busy, measured). More threads cannot issue
  one stream's kernels faster; fewer launches (CUDA graphs) is the lever.
- **Hardware-accelerated GPU scheduling: leave it on.** Off was not faster:
  all-in 70.2 ms against 67.7 ms, Linux config 119.5 against 113.7, default
  185.1 against 183.6. (The first all-in run after the reboot read 75.2 ms;
  its repeat 70.2.)
- `cProfile` attributes GPU waits to whatever Python call blocks on them. Use
  a sampling profiler with native frames for CPU questions.

### Results

Extreme profile, same clip, warm-up 120. "Before" is `5942ed9` as measured
by the harness (which skipped the boot warm-up). The other columns are booted
with `--live-warmup`, the way the live server boots. Hardware GPU scheduling
was off for them; with it on, expect a little better.

| Preset | before | `a6f6b8e` | `d1cd0ea` | + TensorRT (`cf17832`) | fps generated now |
|---|---:|---:|---:|---:|---:|
| Default | 183.6 | 178.2 | 175.4 | — | 5.7 |
| Tiny decoder + INT8 + RIFE graphs | 90.9 | 87.6 | 83.8 | 82.5 | 11.9 / 12.1 |
| All-in + RIFE graphs | 67.7 | 60.8 | 58.5 | 55.6 | 17.1 / 18.0 |
| Tiny VAE + INT8 + flow upscaler + RIFE graphs (production) | 113.7 | 97.8 | 95.0 | 82.8 | 10.5 / 12.1 |

(ms per frame; shown fps is twice the generated fps.)

- The default preset is too slow on this machine; the opt-in presets are
  required.
- Where the production config's first 15.7 ms came from (single runs,
  repeated once): transformer CUDA graphs ~3 ms, tiny VAE in NHWC 6.6 ms,
  compiled upscaler 3.9 ms, no stalled frames ~2.4 ms. Stage graphs, the
  reused timestep results and the skipped text embedder added 2.8 ms.
- Quality of the new defaults against the old switches: LPIPS 0.0130
  (noise floor 0.0130) for the production config, 0.0096 (0.0101) for all-in.
  Everything after `a6f6b8e` on the torch path is bit-identical (whole
  frames, 300/300).
- Memory: about 20 GiB reserved at the peak of the boot warm-up, 12 GiB right
  after it, 13–14 GiB while running.
- RIFE as CUDA graphs is worth about 3 ms here (0.3 ms on Linux).

GPU profile of the current build (profiled frames):

| | wall | GPU kernels | busy | GEMM | conv | attention | gather / copy | elementwise |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All-in | 60.6 | 51.7 | 85% | 24.1 | 8.8 | 9.1 | 5.7 | 3.6 |
| Production | 104.9 | 95.4 | 91% | 33.7 | 32.2 | 16.2 | 8.9 | 3.9 |

The INT8 matmul kernel alone is 21.5 ms of the all-in frame (140 calls).
Both presets are GPU-bound now; what is left is real work at 110 W.

Main-thread time per all-in frame (62.4 ms): everything before the
transformer is issued takes ~4 ms (input 0.3, encode 1.9, change mask 1.0,
indices 0.3), the two graph replays 4.7 ms, decode 1.8 ms, and 47.9 ms are
spent waiting for the GPU at the final download.

### Conv stages in isolation

Each stage alone on the laptop, ms per call. "As it ran" is the setup before
this round.

| Stage | as it ran | fastest found | how |
|---|---:|---:|---|
| Upscaler UNet, 80×144 | 12.6 (eager) | 9.5 | compiled (+ CUDA graph) |
| TAEF2 decode → 1152×640 | 9.4 | 9.5 | no gain alone; NHWC helps in the pipeline |
| TAEF2 decode → 576×320 | 3.7 | 2.2 | NHWC |
| TAEF2 encode 576×320 | 3.1 | 2.1 | NHWC |
| RIFE 1152×640 | 12.6 | 10.9 | CUDA graph (`rife_cudagraphs`) |
| RIFE 576×320 | 5.2 | 3.1 | CUDA graph |
| Full VAE decode 576×320 | 34.1 | 33.1 | none |
| Full VAE encode 576×320 | 16.7 | 14.8 | fp16 + NHWC (not adopted) |

NHWC did nothing for the upscaler, RIFE or the full VAE (slower when
uncompiled). fp16 instead of bf16 stayed within 1 ms. Upscaling to 1152×640
costs about 24 ms of real GPU work per frame: 9.5 ms UNet, 7 ms more decode,
8 ms more RIFE.

### Transformer step as CUDA graphs (`transformer_cudagraphs`)

The step is recorded once per shape and replayed with a single launch
(`StepGraphs` in `transformer_flux2.py`, one per `SpatialCache`). It works
with live input: the per-frame inputs (latents, prompt embedding, timestep,
mask and active-row indices) are copied into the graph's fixed input tensors
before each replay, and the spatial cache is updated in place by the graph.

Measured:

- **Bit-identical.** Every step run both ways from the same cache state,
  live: output and all cache tensors equal in 4,060 of 4,060 steps (plain,
  prompt-travel pattern, reference on, both, reference off again; INT8 + tiny
  VAE and bf16 + full VAE; four revisions of the code).
- **−2.7 to −3.0 ms per frame** on every preset but the default (−1.5 ms),
  separate processes, same frames. One step alone: 23.5 → 22.0 ms at 320
  active rows; the step is mostly real GPU work that scales with the row
  count (about 0.058 ms per row plus 3.5 ms fixed).
- Recording costs 11–15 ms and about 28 MiB per graph with a shared pool.

What the implementation has to respect:

- One graph per (active text rows, active image rows, input shapes, position
  ids); the row bucketing bounds the count.
- Cache tensors keep their memory (in-place `index_copy_` / `copy_`). Masks
  with execute-only rows rebind them and run normally.
- Nothing may wait for the GPU while recording: no block compiling, no Triton
  kernel autotuning, no cuDNN plan being built. A step is recorded on its
  second occurrence and right after running normally on the very objects the
  recording uses. The static mask copy keeps the mask's class, because the
  compiled blocks guard on it (the harness wraps it in a subclass; a base
  class copy forced a recompile inside the recording).
- Everything the graph reads but does not own (position ids, RoPE tables)
  stays referenced by the graph's entry.
- All graphs of one cache share a memory pool and live as long as it.
  Recording into the pool of a destroyed graph fails with an allocator assert.
- **A failed recording is recoverable.** `torch.cuda.graph` leaves its capture
  stream current when `capture_end` raises, and CUDA reports the failure once
  more on the next kernel launch. Switching the stream back and absorbing
  that one error restores normal operation; the step stays on the normal path
  and recording stops after three failures. Tested by injecting a GPU wait
  into two recordings: no crash, the following 300 steps bit-identical.

### Whole-frame identity checks

The INT8 + tiny VAE presets turned out to be deterministic inside one
process: reset the caches, play the same 300 frames again, and every output
frame is bit-identical. That makes whole-frame level-0 proofs cheap: one pass
with a feature forced off, one with it on.

- Old code path (no graphs, timestep results recomputed, text always
  embedded) against the new defaults: 300/300 frames identical, all-in and
  production config.
- Every graph off against transformer graphs, stage graphs, or both: 300/300.
- The bf16 + full VAE default preset is not deterministic even against
  itself (0/300), so there the step-level comparison is the proof.
- Since NHWC + graphs, the all-in preset is also bit-reproducible across
  processes (two separate runs: 300/300 identical).

### Reused work inside the transformer step

- The timestep embedding and the three modulation projections depend only on
  the timestep. The pipeline hands over the same timestep tensor object per
  schedule step and the transformer reuses the last results (four large
  matrix-vector products per step, ~1 ms of GPU time per frame).
- With no active text row the text stream is never read (its keys / values
  come from the spatial cache), so `context_embedder` over all 512 text
  tokens (~0.6 ms per step) is skipped.

### Conv stages as CUDA graphs (`stage_cudagraphs`)

The VAE encoder / decoder and the upscaler UNet have fixed shapes; each is
replayed from one recorded graph (`RecordedModule` in `cuda_graphs.py`).
Bit-identical, about −1 ms on all-in, −0.4 ms with the upscaler.

### Conv stages as TensorRT engines (`conv_backend: "tensorrt"`, opt-in)

Each stage alone, ms per call:

| Stage | torch, compiled | TensorRT fp16 |
|---|---:|---:|
| Upscaler UNet 80×144 | 9.7 | 6.1 |
| RIFE 1152×640 | 11.6 | 6.3 |
| RIFE 576×320 | 4.9 | 2.4 |
| TAEF2 encode 576×320 | 3.1 | 1.2 |

End to end: production config 95.1 → 82.8 ms (−13%), all-in 58.1 → 55.6 ms.
Output against the torch path: LPIPS 0.0116 (noise floor 0.0122) for the
production config; 0.0106 for all-in, the size of the earlier run-to-run
spread (that preset's torch path is now bit-reproducible, so there is no
noise to hide in). Level 1, opt-in because of the dependency:

```
pip install --extra-index-url https://pypi.nvidia.com tensorrt-cu12 onnx
... --set conv_backend=tensorrt
```

- `TrtStage` (`trt_stage.py`) exports a stage once per input shape (ONNX,
  fp16; TensorRT 11 networks are strongly typed, there is no FP16 builder
  flag any more), builds the engine in 20–70 s and caches it under
  `~/.cache/fluxrt/tensorrt`, keyed by stage, shapes, weights, GPU and
  TensorRT version. The first boot takes about 2 min longer; later boots are
  ~20 s shorter than with `torch.compile`.
- Without the packages, or if a build fails, the stage stays on torch.
- Not covered yet: the full VAE (default and tiny-decoder presets keep its
  torch encoder), which spends 33 ms decoding and 17 ms encoding.
- TensorRT warns that it synchronizes when run on the default CUDA stream.
  Running the frame loop on a non-default stream might give a little more.
- First measured in a second, complete venv, then installed into the laptop's
  production venv with the same versions (`tensorrt-cu12` 11.3.0.99, `onnx`
  1.23.1, which also bring `protobuf` and `ml-dtypes`; nothing else changed)
  and measured again: 83.2 ms production config, 55.8 ms all-in.

### Tried: output download overlapped with the next frame (not adopted)

Prototype: frame N's output is downloaded at frame N+1's first GPU sync
instead of at its own end, so the next frame's input / encode / change-mask
work overlaps the GPU tail. Bit-identical (300/300). Gain: 61.7 → 60.5 ms
(all-in), 99.5 → 97.8 ms (production). It changes the live loop's publish
timing for ~2%, and the stage graphs took part of the same slack, so it is
left out.

### Flow upscaler

- The pipeline called `scheduler.set_timesteps(1)` and `scheduler.step` every
  frame, on the scheduler it shares with the main loop; the step looks its
  index up with `.item()`, a GPU sync. The one-step schedule is now computed
  once and the Euler step written out with the same operations in the same
  order: bit-identical (8/8 outputs).
- The UNet was the only network still running uncompiled.

### Prompt travel refresh

48-frame travel, all-in preset, same clip segment and transition:

| | mean | p50 | p95 |
|---|---:|---:|---:|
| `stride` (default) | 151.4 | 120.9 | 226.7 |
| `rolling` | 151.1 | 147.7 | 153.9 |

Rolling removes the sawtooth at the same total time, but mid-travel
neighbouring patches are computed under different prompt blends: a fine
dither on flat areas and a flatter tone on big transitions. It stays opt-in.

### Full stack on the laptop: where the time and the CPU go (2 Oct 2026)

Live, with the kiosk connected (FluxRT alone on the GPU, 92 W):

| Seconds after a prompt change | ms per frame |
|---|---|
| 0–12 | 213–235 |
| 12–20 | 79–96 (the bench figure) |

A morph costs about 2.7 frames' worth, and its length was counted in
generated frames sized from `fps_pipeline` (one frame's 1 / time): a 4 s
morph ran 12 s of every 20. Now `prompt-travel:<n>s:…` is timed on the clock
and `/healthz` has `fps_pipeline_avg`. Re-measure after the next restart.

py-spy on the live server (`record --nonblocking`, read-only):

- Model process, one core at 100%: 89% of samples in the output download
  (`.cpu()` in `interpolate_frames`), i.e. CUDA's spinning wait. `gpu_wait:
  "sleep"` replaces it with a sleeping wait on a CUDA event.
- Server process, 1.7 cores: VP8 encode of the output 34%, input decode +
  BGR conversion + crop/resize 28% (30 input frames a second for an engine
  that takes about 10; the kiosk now caps its uplink at 1.5 × the generated
  fps), frame pumps 12%.

The GPU's power limit sat at 92 W of 110 W: the 15 W Dynamic Boost goes to
the GPU only while the CPU is idle, and about six cores were busy.

## Tried and dropped

| Idea | Result |
|---|---|
| Compiler `max-autotune` | About 1 ms, worse p95, minutes of extra warm-up |
| FP8 (e4m3) matmuls | 2× bf16 at three times INT8's error; INT8 is 3×. Row-wise FP8 slower than bf16 |
| Whole-graph compile | Within 3 ms of per-block, and stalls live sessions on every new variant |
| SageAttention, flash-attn on Windows | See above |
| Hardware GPU scheduling off | 3–5% slower on the fast presets |
| `active_row_bucket=32` | Same mean, twice the warm-up and more cached memory |
| Synthetic plan / shape priming at startup | Did not prevent the first-use stall; removed |
| NHWC for upscaler UNet, RIFE, full VAE | No gain or slower |
| fp16 instead of bf16 for conv stages | Within 1 ms |
| `prompt_travel_refresh=rolling` as default | Visible dither mid-travel |
| Host-side masks | No gain on the laptop (67.7 against 68.0 ms); kept, harmless |
| Hybrid TAEF2 decoder without `image_processor.postprocess` | Garbled output (LPIPS 0.48); a bug, fixed |

## Wrong turns

Worth keeping, because each looked convincing at the time:

- The laptop stalls were blamed first on INT8 recompiles, then on the garbage
  collector. Both wrong: it was the per-shape cuDNN plan build. The "growing
  slow gc" log lines were an artifact of `torch.profiler`.
- "A laptop 5090 is 70–90% of a 4090" assumed a 150–175 W chassis. This one is
  110 W.
- A stale compile cache made a new attention backend look ineffective.
- "CUDA graphs will take all-in from 60 to 40 ms" assumed the transformer step
  was mostly launch overhead. Measured: mostly real GPU work; the gain is
  about 7 ms, not 20.
- Bucketing plus synthetic priming looked like the complete stall fix because
  the harness's untimed warm-up frames absorbed most first-use stalls. The
  per-frame data still showed three, always at the same frames.
- A per-frame "CPU profile" from a sampling profiler overstates CPU work in a
  GPU-bound loop: the sampler's pauses hide the GPU waits. Timers around the
  stages gave the real split.
- "The fast presets are CPU-bound on Windows" was half right. The CPU share is
  large (launches, cuBLAS call setup, Python), but most of it overlaps GPU
  work that has to run anyway.

## Open

- TensorRT for the full VAE; the frame loop on a non-default CUDA stream.
- Whether `conv_backend` should become `"auto"` (TensorRT when installed).
- One benchmark from an interactive desktop session (all runs so far came
  from a non-interactive SSH session), and with hardware GPU scheduling on.
- NVIDIA "Prefer No Sysmem Fallback" setting.
- `/offer` output is capped at 30 fps (`FluxRTTrack(fps=30)`), below what
  the fast presets generate on Linux.
- The first prompt travel after boot showed two ~480 ms stalls in one test
  (before the warm-up sweep); re-check.

## Boot commands

Windows show machine (venv, flags and the show config in one script):

```powershell
scripts\start_show.ps1                      # production: flow upscaler, 1152x640 out
scripts\start_show.ps1 -Fast                # + mask_dilation=1
scripts\start_show.ps1 -Fast -NoUpscaler    # fastest, 576x320 out
scripts\start_show.ps1 -Fast --set gpu_wait=sleep   # extra arguments go to run_webrtc.py
```

Settings for every start are in `configs/show_laptop.json` (`int8_linear`,
`rife_cudagraphs`, `conv_backend`, `gpu_wait`, lip transfer off).

```bash
# default, no quality cost
python scripts/run_webrtc.py --config configs/config_with_reference.json --interp 1 --no-server-camera

# tiny decoder + INT8 (3.15× on the 4090, subtle softening)
... --set vae_decoder=taef2 --set int8_linear=true

# fast, full VAE (2.55×)
... --set int8_linear=true --set mask_dilation=1

# all-in (4.35×, tiny VAE look)
... --tiny-vae --set int8_linear=true --set mask_dilation=1 --set rife_cudagraphs=true

# Linux production config
python scripts/run_webrtc.py --no-server-camera --tiny-vae --flow-upscaler --interp 1 \
  --config configs/config_with_reference.json --set int8_linear=true --set rife_cudagraphs=true
```
