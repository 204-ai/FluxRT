# Testing `claude/happy-mayer-jrgkt5` on the show laptop (`204-eeg`)

For a Claude Code session (or a person) at the MSI Stealth A16 show laptop.
The branch changes the CPU side of the live path: shared memory, the output
scheduler, the WebRTC input and output. It was measured on a 4-core cloud VM
without a GPU, with a stub in place of the model (`docs/perf-notes.md`,
"WebRTC ingress and egress"). This is the first run on real hardware.

The machine's own runbook is `C:\AI\realtime-client\OPS-show-machine.md`
(layout, presets, how the stack starts); the rules for working on a show
laptop are in `C:\AI\realtime-client\SETUP-agent-brief.md`. Read both first.

## What the branch changes, and what applies on this machine

The kiosk on this laptop talks to FluxRT over the **local transport**
(`ws://localhost:8765/ws/local`, raw frames), not WebRTC. So:

| Change | Show path (kiosk, local transport) | WebRTC viewers / benchmark | Model process |
|---|---|---|---|
| Output scheduler: every generated frame shown when frame times vary | yes | yes | — |
| Shared memory seqlock (no torn frames) | yes | yes | input read, warm-up writes |
| Output pump publishes to waiting consumers instead of polling | yes | yes | — |
| I420 once per frame for all viewers (`egress_i420`) | no (converted only while a WebRTC track uses it) | yes | — |
| VP8 denoiser off (`FLUXRT_VP8_*`) | no | yes | — |
| Inbound video decoded on its last packet (`FLUXRT_JITTER_MARKER`) | no | yes | — |

What to find out here:

1. The model path is unchanged: same ms per frame, output inside the noise.
2. The WebRTC side gains what it gained on the VM, on this CPU.
3. The live show (kiosk, local transport) does not regress: FluxRT CPU,
   frames published and sent, fps, GPU power limit; and whether the
   scheduler fix shows in frames published per generated frame.
4. Nothing breaks on Windows (tests, shared memory, Spout if used).

## Rules (from the agent brief, plus this test)

- It is the show machine. **Do not stop or restart a running FluxRT, the
  kiosk or Chrome without asking the user.** Steps 3 and 5 need the GPU to
  themselves; ask before each.
- **Do not touch the production checkout `C:\AI\FluxRT`** (no checkout, no
  pull, no reset) or its venv. The branch and the comparison base go into
  separate worktrees; the CPU-only tests get their own venv.
- Python only in a venv. Scripts ASCII only (PowerShell 5.1 reads a `.ps1`
  without BOM as ANSI).
- For measurements: MSI profile **Extreme** (check
  `nvidia-smi --query-gpu=enforced.power.limit,power.max_limit --format=csv`:
  the limit should be 92-110 W), lid open, Chrome / Spotify / Task Manager /
  RustDesk closed unless the step says otherwise.
- Say what you changed and how to undo it. Everything below is undone by
  section 7.

## 0. Preflight (read only)

```powershell
git -C C:\AI\FluxRT status --short
git -C C:\AI\FluxRT log --oneline -1
git -C C:\AI\FluxRT branch --show-current
try { Invoke-RestMethod http://localhost:8765/healthz -TimeoutSec 3 | Select-Object ready, peers, fps_pipeline_avg } catch { "FluxRT not running" }
Get-Process chrome -ErrorAction SilentlyContinue | Measure-Object | Select-Object Count
nvidia-smi --query-gpu=name,driver_version,enforced.power.limit,power.max_limit,memory.used --format=csv
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } | Select-Object InterfaceAlias, IPAddress
```

Write the results at the top of the report (section 6). The last line must
list at least one address: WebRTC between two processes on this machine needs
a network interface with an address (`OPS-show-machine.md`), so step 4 needs
WiFi or a cable.

## 1. Worktrees

FluxRT is public; fetching needs no credentials.

```powershell
cd C:\AI\FluxRT
git fetch origin claude/happy-mayer-jrgkt5
git worktree add --detach C:\AI\FluxRT-branch FETCH_HEAD
git worktree add --detach C:\AI\FluxRT-main 974bdb4
```

`974bdb4` is the commit the branch starts from: the comparison isolates the
branch's changes from whatever else is deployed. If the fetch fails, ask the
user for a git bundle of the branch (`git clone x.bundle`).

The model folders and the venv live only in `C:\AI\FluxRT`. Link them into
both worktrees (junctions need no administrator; nothing is copied):

```powershell
foreach ($wt in 'C:\AI\FluxRT-branch', 'C:\AI\FluxRT-main') {
  foreach ($d in '.venv', 'FLUX.2-klein-4B', 'FLUX.2-klein-4B-int8', 'RIFE-safetensors', 'taef2', 'FlowUpscaler', 'LivePortrait', 'local_samples') {
    if ((Test-Path "C:\AI\FluxRT\$d") -and -not (Test-Path "$wt\$d")) { New-Item -ItemType Junction -Path "$wt\$d" -Target "C:\AI\FluxRT\$d" | Out-Null; "linked $wt\$d" }
  }
}
```

**The production venv has FluxRT installed editable, pointing at
`C:\AI\FluxRT\src`.** A worktree's `scripts\run_webrtc.py` would import the
production code unless `PYTHONPATH` names the worktree's `src`. Every live
start below sets it, and section 5 checks the log to prove which code runs.
`perf_ab.py --repo` and `perf_webrtc.py --repo` put the right `src` first
themselves.

## 2. CPU-only checks: tests and the WebRTC stage benchmark (~5 min)

A separate venv with the WebRTC packages at the production venv's versions:

```powershell
uv pip freeze --python C:\AI\FluxRT\.venv\Scripts\python.exe | Select-String -Pattern '^(aiortc|av|opencv-python|opencv-python-headless|numpy|fastapi|uvicorn|httpx|pillow|python-multipart)=='
```

```powershell
uv venv C:\AI\fluxrt-bench-venv --python 3.12
uv pip install --python C:\AI\fluxrt-bench-venv\Scripts\python.exe <the pinned lines from above> psutil scikit-image pytest pytest-asyncio requests
```

(If the production venv has `opencv-python`, pin that; `scikit-image` only
supplies the photographic test content.)

```powershell
$out = "$env:USERPROFILE\fluxrt-setup\branch-test"
New-Item -ItemType Directory -Force $out | Out-Null
cd C:\AI\FluxRT-branch
$bench = 'C:\AI\fluxrt-bench-venv\Scripts\python.exe'
& $bench -m pytest tests -q --ignore=tests\webrtc
cd tests\webrtc; & $bench -m pytest -q; cd ..\..
& $bench scripts\test_whep.py
& $bench scripts\test_whip.py
& $bench scripts\perf_webrtc.py stages | Tee-Object "$out\stages.md"
```

Run the two test folders separately, as above: `tests\webrtc` has its own
`pytest.ini` (asyncio mode), and run from the top its async tests fail.
Expected: every test passes (on the VM: 76 + 130 unit tests, WHEP 27 checks,
WHIP 19 checks). `stages` prints four tables: egress conversion, VP8 settings,
ingress, shared-memory tearing. Keep its whole output for the report.

These tests are the first run of the seqlock on Windows (named shared memory
there is page-rounded and freed with its last handle); a failure in
`tests\test_shared_tensor.py` or `tests\test_output_scheduler.py` is the
important one to report.

## 3. Model path unchanged: `perf_ab` (~45 min, GPU to itself)

Ask the user before stopping a running FluxRT and Chrome. Then use this
machine's bench script, which runs the presets of `OPS-show-machine.md` with
the same clip, warm-up and GPU log:

```powershell
cd C:\AI\realtime-client\scripts\windows
powershell -ExecutionPolicy Bypass -File bench-fluxrt.ps1 -Repo C:\AI\FluxRT-main -Presets production-trt,allin-trt -Tag _main
powershell -ExecutionPolicy Bypass -File bench-fluxrt.ps1 -Repo C:\AI\FluxRT-main -Presets production-trt,allin-trt -Tag _main2
powershell -ExecutionPolicy Bypass -File bench-fluxrt.ps1 -Repo C:\AI\FluxRT-branch -Presets production-trt,allin-trt -Tag _branch
```

Each writes `%USERPROFILE%\fluxrt-setup\bench<Tag>.log` (last line
`BENCH DONE`) and `runs\<preset><Tag>\`. Run them one after another, never
two at once (a tag without a leading `-`: PowerShell could read `-main` as a
parameter name). If `bench-fluxrt.ps1` is not in this checkout of
realtime-client, run the equivalent by hand per preset (production-trt shown;
allin-trt replaces `enable_flow_upscaler=true` with `mask_dilation=1`):

```powershell
cd C:\AI\FluxRT-branch
.\.venv\Scripts\python.exe scripts\perf_ab.py run --repo C:\AI\FluxRT-branch --config C:\AI\FluxRT-branch\configs\show_laptop.json --set interpolation_exp=1 --set area_downscale=false --set enable_tiny_vae=true --set int8_linear=true --set enable_flow_upscaler=true --set rife_cudagraphs=true --set conv_backend=tensorrt --clip C:\AI\test-clip-webcam.mov --frames 300 --warmup 120 --live-warmup --out $env:USERPROFILE\fluxrt-setup\runs\production-trt_branch
```

Compare (per preset; `--lpips` only if the `lpips` package is in the venv):

```powershell
cd $env:USERPROFILE\fluxrt-setup
C:\AI\FluxRT\.venv\Scripts\python.exe C:\AI\FluxRT-branch\scripts\perf_ab.py compare runs\production-trt_main runs\production-trt_branch --noise runs\production-trt_main2 --out branch-test\report-production
C:\AI\FluxRT\.venv\Scripts\python.exe C:\AI\FluxRT-branch\scripts\perf_ab.py compare runs\allin-trt_main runs\allin-trt_branch --noise runs\allin-trt_main2 --out branch-test\report-allin
```

Expected: ms per frame (mean, p50, p95) of the branch inside the spread of
the two main runs (two runs of the same setting differed by ~0.1 ms on these
machines), output differences inside the noise floor. The model process only
changed how it reads its input from shared memory and how the warm-up writes
it. `perf_ab` drives the model in-process: it does not exercise the output
scheduler (step 5 does).

## 4. WebRTC loopback on this CPU (~12 min, GPU not used)

FluxRT need not be stopped for this one, but nothing else should load the CPU
(it competes with the client, server and stub). Ask the user; with the show
running the numbers are worse for both sides alike.

```powershell
$out = "$env:USERPROFILE\fluxrt-setup\branch-test"
cd C:\AI\FluxRT-branch
$bench = 'C:\AI\fluxrt-bench-venv\Scripts\python.exe'
& $bench scripts\perf_webrtc.py loopback --repo C:\AI\FluxRT-main --label old --set "bench_frame_pattern=[1.0]" --out $out\A_old.json
& $bench scripts\perf_webrtc.py loopback --repo C:\AI\FluxRT-branch --label new --set "bench_frame_pattern=[1.0]" --out $out\A_new.json
& $bench scripts\perf_webrtc.py loopback --repo C:\AI\FluxRT-main --label old-vary --out $out\C_old.json
& $bench scripts\perf_webrtc.py loopback --repo C:\AI\FluxRT-branch --label new-vary --out $out\C_new.json
& $bench scripts\perf_webrtc.py loopback --repo C:\AI\FluxRT-main --label old-15fps --input-fps 15 --set "bench_frame_pattern=[1.0]" --out $out\E_old.json
& $bench scripts\perf_webrtc.py loopback --repo C:\AI\FluxRT-branch --label new-15fps --input-fps 15 --set "bench_frame_pattern=[1.0]" --out $out\E_new.json
& $bench scripts\perf_webrtc.py compare $out\A_old.json $out\A_new.json $out\C_old.json $out\C_new.json $out\E_old.json $out\E_new.json | Tee-Object "$out\loopback.md"
```

After the runs, check no stub or scheduler process was left behind
(`Get-Process python` should show only what ran before). On the VM: latency
p50 276 -> 235 ms (A), generated frames shown 54% -> 94% (C), 326 -> 239 ms
(E), server CPU -6 to -16%. If a run reports `client failed`, the most likely
cause is no network address (preflight).

## 5. Live show, old and new (~30 min, needs the user at the laptop)

The comparison is the base commit (`FluxRT-main`) and the branch, each
started the way the show starts (`start_show.ps1`, same flags), with the
kiosk on the webcam clip. Ask the user before stopping the running server.
Each side in its own fresh PowerShell window.

Branch:

```powershell
$env:PYTHONPATH = 'C:\AI\FluxRT-branch\src'
C:\AI\FluxRT-branch\scripts\start_show.ps1
```

Old side:

```powershell
$env:PYTHONPATH = 'C:\AI\FluxRT-main\src'
C:\AI\FluxRT-main\scripts\start_show.ps1
```

**Check which code runs:** the branch's startup log shows
`VP8 encoder: libvpx {'noise-sensitivity': '0'}` and
`jitter buffer: video frames decode on their last packet`; the old side's
does not. If the branch's log lacks them, the production code is running
(PYTHONPATH not set in that window): stop and fix before measuring. The
deployed `C:\AI\start-fluxrt.ps1` may be at another commit than `974bdb4`;
use it only as a third data point.

When `/healthz` says `"ready": true`, the user opens the kiosk
(`cd C:\AI\realtime-client; yarn preview`, Chrome), picks
**Device > Video file...** `C:\AI\test-clip-webcam.mov` in the admin
(Ctrl+Shift+K), and leaves it on one prompt set for the measurement. Then
measure 60 s:

```powershell
$a = Invoke-RestMethod http://localhost:8765/healthz
$cpu0 = Get-Process python | Measure-Object CPU -Sum
Start-Sleep 60
$b = Invoke-RestMethod http://localhost:8765/healthz
$cpu1 = Get-Process python | Measure-Object CPU -Sum
"published frames/s: {0:N1}" -f (($b.output_frames - $a.output_frames) / 60)
"generated fps (avg): {0}" -f $b.fps_pipeline_avg
"FluxRT CPU, cores: {0:N2}" -f (($cpu1.Sum - $cpu0.Sum) / 60)
$b.local_clients | ConvertTo-Json
$b.gpu | ConvertTo-Json
```

(`Get-Process python` sums every python process: make sure only FluxRT's run.)
Also note the kiosk's dev overlay (D): `<- engine` fps, `published`, `lag`.
Do the same for the old side, same prompt set, same clip.

What to look for: published frames per generated frame close to 2 (RIFE x2)
on the branch, possibly below 2 on the old side when frame times jump
(prompt changes, motion: that is the scheduler fix); FluxRT CPU not higher
than before (the local transport path does no I420 work now); enforced GPU
limit and generated fps not lower.

Optional, if time allows:

- The kiosk on WebRTC instead of the local transport (admin: the engine's
  transport `webrtc`, engine restart): the branch's WebRTC gains on the live
  path. Same 60 s measurement.
- Spout to TouchDesigner, if the show uses it:
  `start_show.ps1 --spout-in TDOut --spout-out FluxRT`, check the output
  arrives (the Spout sender now waits on the publisher; never run before).

## 6. Report

Write `%USERPROFILE%\fluxrt-setup\branch-test\REPORT.md` with: the preflight
output, the `stages` tables, the two `perf_ab compare` summaries (speed table
and the noise / difference lines), the `perf_webrtc compare` table, the live
numbers for old and new, and any failure with its traceback. No GitHub
credentials on this box: the user carries the report back (paste it into the
session that asked for the test).

## 7. Put everything back

- Stop the branch server (Ctrl+C). If the user wants the show running again:
  `C:\AI\start-fluxrt.ps1` in a fresh shell (no PYTHONPATH).
- Remove the worktrees (junctions go with them, their targets are untouched):
  `git -C C:\AI\FluxRT worktree remove --force C:\AI\FluxRT-branch` and the
  same for `C:\AI\FluxRT-main`. Check `C:\AI\FluxRT\FLUX.2-klein-4B` still
  exists afterwards.
- `C:\AI\fluxrt-bench-venv` can be deleted (`Remove-Item -Recurse`), or kept
  for the next test; say which.
- The bench runs stay in `%USERPROFILE%\fluxrt-setup\runs` (safe to delete).
  The compile and TensorRT caches under `%USERPROFILE%\.cache\fluxrt` were
  shared and stay valid for the production checkout.
