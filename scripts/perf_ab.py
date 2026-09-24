"""A/B harness: same input clip, same config, two code revisions or two config
toggles -> hard numbers (per-frame processing time) and a visual comparison.

Drives the LIVE path in-process, one fixed input frame per iteration (no
free-running loop, no WebRTC, no drain-to-latest), so both sides see exactly
the same frame sequence and the spatial cache evolves identically.

    # 1. one run per side (the server must be stopped: two models don't fit and
    #    a shared GPU skews the timings)
    python scripts/perf_ab.py run --repo /path/to/FluxRT      --clip clip.mp4 --out runs/main
    python scripts/perf_ab.py run --repo /path/to/FluxRT-perf --clip clip.mp4 --out runs/perf
    python scripts/perf_ab.py run --repo /path/to/FluxRT-perf --clip clip.mp4 --out runs/perf-novae \\
        --set compile_vae=false

    # 2. noise floor: the same side twice (GPU kernels are not bit-deterministic
    #    run to run) — a diff at this level is not a change
    python scripts/perf_ab.py run --repo /path/to/FluxRT --clip clip.mp4 --out runs/main2

    # 3. compare
    python scripts/perf_ab.py compare runs/main runs/perf --out reports/main-vs-perf

--repo is the checkout whose code runs (models resolve relative to it, like the
server). The harness itself only needs to exist in the checkout you launch it
from; revisions without step_live() are driven by an equivalent legacy loop.
"""

import argparse
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time

import cv2
import numpy as np


def _parse_override(item):
    key, _, raw = item.partition("=")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key, value


def _git(repo, *args):
    try:
        return subprocess.check_output(["git", "-C", repo, *args], text=True).strip()
    except Exception:
        return None


def _load_clip(path, count, height, width, crop):
    """First `count` frames of the clip (looped if shorter), cropped/resized to
    the model resolution with the repo's own crop_maximal_rectangle."""
    cap = cv2.VideoCapture(path)
    source = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        source.append(frame)
    cap.release()
    if not source:
        sys.exit(f"no frames read from {path}")
    return [crop(source[i % len(source)], height, width) for i in range(count)]


def _legacy_step(sub):
    """The live-loop body of revisions before step_live() existed."""
    sub.update_process_state()
    sub._advance_prompt_travel()
    frame = sub.input_shared_tensor.to_numpy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if hasattr(sub, "process_frame_to_gpu"):  # perf/hot-path 1046a43
        out = sub.process_frame_to_gpu(frame)
    else:  # main
        out = sub.convert_np_to_torch(sub.process_frame_with_pipeline(frame))
    return sub.interpolate_frames(out)


def cmd_run(args):
    repo = os.path.abspath(args.repo)
    os.chdir(repo)  # model paths in the config are relative to the repo root
    sys.path.insert(0, os.path.join(repo, "src"))

    import torch
    from multiprocessing import Value
    from fluxrt.utils.shared_tensor import SharedTensor
    from fluxrt.utils import crop_maximal_rectangle
    from fluxrt.stream_processor.model_inference_subprocess import (
        ModelInferenceSubprocess,
    )

    with open(args.config) as f:
        config = json.load(f)
    for item in args.set:
        key, value = _parse_override(item)
        config[key] = value
    if args.prompt is not None:
        config["default_prompt"] = args.prompt
    config["batch_mode"] = False
    config["logging"] = False

    height, width = config["resolution"]["height"], config["resolution"]["width"]
    out_h, out_w = height, width
    if config.get("enable_flow_upscaler", False):
        out_h, out_w = height * 2, width * 2
    exp = config.get("interpolation_exp", 1)

    crop_kwargs = {}
    if "area_downscale" in inspect.signature(crop_maximal_rectangle).parameters:
        crop_kwargs["area_downscale"] = bool(config.get("area_downscale", True))
    frames_in = _load_clip(
        args.clip, args.warmup + args.frames, height, width,
        lambda fr, h, w: crop_maximal_rectangle(fr, h, w, **crop_kwargs),
    )

    input_tensor = SharedTensor((height, width, 3), create=True)
    output_tensor = SharedTensor((2**exp, out_h, out_w, 3), create=True)
    sub = ModelInferenceSubprocess(
        config, input_tensor.name, output_tensor.name, Value("b", False), Value("f", 0.0)
    )
    try:
        t_load = time.perf_counter()
        sub.process_init()
        load_s = time.perf_counter() - t_load
        step = sub.step_live if hasattr(sub, "step_live") else (lambda: _legacy_step(sub))

        outputs = np.empty((args.frames, out_h, out_w, 3), dtype=np.uint8)
        ms = []
        warmup_ms = []
        for i, frame in enumerate(frames_in):
            np.copyto(input_tensor.array, frame)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pack = step()  # CPU uint8: the download already synchronized
            dt = (time.perf_counter() - t0) * 1000.0
            if i < args.warmup:
                warmup_ms.append(dt)
                continue
            ms.append(dt)
            outputs[i - args.warmup] = pack[-1]  # the generated frame (last in the pack)
            if (i - args.warmup) % 50 == 0:
                print(f"frame {i - args.warmup}/{args.frames}: {dt:.1f} ms", flush=True)
    finally:
        input_tensor.close_and_unlink()
        output_tensor.close_and_unlink()
        sub._manager.shutdown()

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "frames.npy"), outputs)
    with open(args.clip, "rb") as f:
        clip_sha = hashlib.sha1(f.read()).hexdigest()
    meta = {
        "label": args.label or os.path.basename(os.path.normpath(args.out)),
        "repo": repo,
        "git_rev": _git(repo, "rev-parse", "--short", "HEAD"),
        "git_branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git(repo, "status", "--porcelain", "--untracked-files=no")),
        "step": "step_live" if hasattr(sub, "step_live") else "legacy",
        "config": config,
        "overrides": args.set,
        "clip": os.path.abspath(args.clip),
        "clip_sha1": clip_sha,
        "frames": args.frames,
        "warmup": args.warmup,
        "load_s": round(load_s, 1),
        "warmup_ms": [round(x, 2) for x in warmup_ms],
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "max_reserved_mb": torch.cuda.max_memory_reserved() // (1024 * 1024),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump({"meta": meta, "ms": [round(x, 3) for x in ms]}, f, indent=1)
    s = _stats(ms)
    print(f"{meta['label']}: mean {s['mean']:.1f} ms  p50 {s['p50']:.1f}  p95 {s['p95']:.1f}  "
          f"-> {s['fps']:.2f} fps generated  ({args.out})")


def _stats(ms):
    a = np.asarray(ms)
    return {
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "max": float(a.max()),
        "fps": float(1000.0 / a.mean()),
    }


def _load_run(path):
    with open(os.path.join(path, "run.json")) as f:
        run = json.load(f)
    run["frames"] = np.load(os.path.join(path, "frames.npy"), mmap_mode="r")
    return run


def _psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def _diff_vis(a, b, gain):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=-1)
    d = np.clip(d * gain, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)


def _label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _quality(fa, fb, n, lpips_net):
    """Output difference of two runs, frame by frame. Pixel PSNR alone is
    misleading here: the same code run twice already differs in fine texture
    (GPU nondeterminism amplified through the spatial cache and 2 steps), so
    also measure structure (PSNR at 1/4 resolution), perceptual distance
    (LPIPS) and flicker, and read every number against the noise floor."""
    per_frame = []
    for i in range(n):
        a, b = np.asarray(fa[i]), np.asarray(fb[i])
        d = np.abs(a.astype(np.int16) - b.astype(np.int16))
        small = [cv2.resize(x, (x.shape[1] // 4, x.shape[0] // 4), interpolation=cv2.INTER_AREA) for x in (a, b)]
        per_frame.append({
            "max": int(d.max()),
            "mean": float(d.mean()),
            "changed_px_pct": float((d.max(axis=-1) > 2).mean() * 100),
            "psnr": _psnr(a, b),
            "psnr_quarter": _psnr(*small),
        })
    psnrs = np.array([p["psnr"] for p in per_frame])
    finite = psnrs[np.isfinite(psnrs)]
    q = {
        "identical": int(np.sum(~np.isfinite(psnrs))),
        "first_diff": next((i for i, p in enumerate(per_frame) if p["max"] > 0), None),
        "psnr_min": float(finite.min()) if len(finite) else float("inf"),
        "psnr_median": float(np.median(finite)) if len(finite) else float("inf"),
        "psnr_quarter_median": float(np.median([p["psnr_quarter"] for p in per_frame])),
        "max_diff_worst": max(p["max"] for p in per_frame),
        "mean_abs_diff": float(np.mean([p["mean"] for p in per_frame])),
        "per_frame": per_frame,
    }
    if lpips_net is not None:
        import torch

        vals = []
        with torch.no_grad():
            for i in range(n):
                t = [torch.from_numpy(np.ascontiguousarray(x[i][..., ::-1])).permute(2, 0, 1)[None]
                     .float().div(127.5).sub(1).cuda() for x in (fa, fb)]
                vals.append(float(lpips_net(*t)))
        q["lpips_mean"] = float(np.mean(vals))
        q["lpips_p95"] = float(np.percentile(vals, 95))
    return q


def _flicker(frames, n):
    """Mean |frame_t - frame_t-1| (0-255): how much the output moves frame to frame."""
    return float(np.mean([np.abs(np.asarray(frames[i]).astype(np.int16) - np.asarray(frames[i - 1]).astype(np.int16)).mean()
                          for i in range(1, n)]))


def cmd_compare(args):
    A, B = _load_run(args.a), _load_run(args.b)
    ma, mb = A["meta"], B["meta"]
    warnings = []
    for key in ("clip_sha1", "frames", "warmup"):
        if ma[key] != mb[key]:
            warnings.append(f"{key} differs: {ma[key]} vs {mb[key]}")
    for key in ("resolution", "default_prompt", "default_seed", "default_steps", "interpolation_exp"):
        if ma["config"].get(key) != mb["config"].get(key):
            warnings.append(f"config.{key} differs: {ma['config'].get(key)} vs {mb['config'].get(key)}")
    fa, fb = A["frames"], B["frames"]
    n = min(len(fa), len(fb))

    lpips_net = None
    if args.lpips:
        import lpips

        lpips_net = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    q = _quality(fa, fb, n, lpips_net)
    per_frame = q["per_frame"]
    qn = None
    if args.noise:
        N = _load_run(args.noise)
        if N["meta"]["clip_sha1"] != ma["clip_sha1"]:
            warnings.append("noise run used a different clip")
        qn = _quality(fa, N["frames"], min(n, len(N["frames"])), lpips_net)
    flicker_a, flicker_b = _flicker(fa, n), _flicker(fb, n)

    sa, sb = _stats(A["ms"]), _stats(B["ms"])
    os.makedirs(args.out, exist_ok=True)

    # contact sheet: rows = sampled frames; A | B | amplified |A-B|
    rows = []
    for i in np.linspace(0, n - 1, args.sheet_frames).astype(int):
        a, b = np.asarray(fa[i]), np.asarray(fb[i])
        rows.append(np.hstack([
            _label(a, f"A {ma['label']} #{i}"),
            _label(b, f"B {mb['label']} #{i}"),
            _label(_diff_vis(a, b, args.gain), f"|A-B| x{args.gain}  max {per_frame[i]['max']}"),
        ]))
    cv2.imwrite(os.path.join(args.out, "sheet.png"), np.vstack(rows))

    h, w = fa.shape[1:3]
    vw = cv2.VideoWriter(os.path.join(args.out, "side_by_side.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps, (w * 3, h))
    for i in range(n):
        a, b = np.asarray(fa[i]), np.asarray(fb[i])
        vw.write(np.hstack([_label(a, "A"), _label(b, "B"), _diff_vis(a, b, args.gain)]))
    vw.release()

    speedup = sa["mean"] / sb["mean"]
    lines = [
        f"# {ma['label']} vs {mb['label']}",
        "",
        f"- A: `{ma['repo']}` @ {ma['git_rev']} ({ma['git_branch']}{', dirty' if ma['git_dirty'] else ''}) "
        f"overrides {ma['overrides'] or 'none'}",
        f"- B: `{mb['repo']}` @ {mb['git_rev']} ({mb['git_branch']}{', dirty' if mb['git_dirty'] else ''}) "
        f"overrides {mb['overrides'] or 'none'}",
        f"- clip `{os.path.basename(ma['clip'])}` ({ma['clip_sha1'][:10]}), {n} measured frames after "
        f"{ma['warmup']} warmup, {ma['config']['resolution']['width']}x{ma['config']['resolution']['height']}, "
        f"{ma['config']['default_steps']} steps, seed {ma['config']['default_seed']}",
        f"- {ma['gpu']}, torch {ma['torch']}",
        "",
        "## Speed (per generated frame, live path)",
        "",
        "| | mean ms | p50 | p95 | min | max | fps generated | VRAM reserved MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| A | {sa['mean']:.1f} | {sa['p50']:.1f} | {sa['p95']:.1f} | {sa['min']:.1f} | {sa['max']:.1f} | "
        f"{sa['fps']:.2f} | {ma['max_reserved_mb']} |",
        f"| B | {sb['mean']:.1f} | {sb['p50']:.1f} | {sb['p95']:.1f} | {sb['min']:.1f} | {sb['max']:.1f} | "
        f"{sb['fps']:.2f} | {mb['max_reserved_mb']} |",
        "",
        f"**B is {speedup:.2f}x the speed of A** ({sa['mean'] - sb['mean']:.1f} ms less per frame).",
        "",
        "## Output difference (A = reference)",
        "",
        "| metric | B vs A |" + (" A vs A again (noise floor) |" if qn else "") ,
        "|---|---:|" + ("---:|" if qn else ""),
    ]
    def row(name, key, fmt):
        return f"| {name} | {fmt(q[key])} |" + (f" {fmt(qn[key])} |" if qn else "")
    lines += [
        row("bit-identical frames", "identical", lambda v: f"{v}/{n}"),
        row("PSNR full res, median (dB)", "psnr_median", lambda v: f"{v:.1f}"),
        row("PSNR full res, worst frame (dB)", "psnr_min", lambda v: f"{v:.1f}"),
        row("PSNR at 1/4 res, median (dB) — structure/colour", "psnr_quarter_median", lambda v: f"{v:.1f}"),
        row("mean abs pixel diff (0-255)", "mean_abs_diff", lambda v: f"{v:.2f}"),
    ]
    if "lpips_mean" in q:
        lines += [
            row("LPIPS mean (lower = closer; ~0.1 visible)", "lpips_mean", lambda v: f"{v:.4f}"),
            row("LPIPS p95", "lpips_p95", lambda v: f"{v:.4f}"),
        ]
    lines += [
        "",
        f"Flicker (mean |frame t - frame t-1|, 0-255): A {flicker_a:.2f}, B {flicker_b:.2f}.",
    ]
    lines += [
        "",
        "Compare against a noise-floor run (the same side twice): GPU kernels are not "
        "bit-deterministic run to run, and the spatial cache carries small differences forward.",
        "",
        f"![sheet](sheet.png)  — `side_by_side.mp4`: A | B | |A-B| x{args.gain}",
    ]
    if warnings:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in warnings]
    report = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(report)
    with open(os.path.join(args.out, "compare.json"), "w") as f:
        json.dump({"a": ma, "b": mb, "speed_a": sa, "speed_b": sb, "speedup": speedup,
                   "quality": q, "noise_floor": qn, "flicker": {"a": flicker_a, "b": flicker_b},
                   "warnings": warnings}, f, indent=1)
    print(report)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)

    r = sp.add_parser("run", help="render a fixed clip through the live path and time it")
    r.add_argument("--repo", default=".", help="checkout whose code and models run")
    r.add_argument("--config", default="configs/stream_processor_config.json",
                   help="config JSON (relative to --repo)")
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, JSON value (repeatable), e.g. compile_vae=false")
    r.add_argument("--prompt", default=None)
    r.add_argument("--clip", required=True)
    r.add_argument("--frames", type=int, default=300)
    r.add_argument("--warmup", type=int, default=40,
                   help="untimed frames first (torch.compile + cache warm-up)")
    r.add_argument("--out", required=True)
    r.add_argument("--label", default=None)

    c = sp.add_parser("compare", help="speed + visual diff of two runs")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out", required=True)
    c.add_argument("--sheet-frames", type=int, default=6)
    c.add_argument("--gain", type=int, default=8, help="diff amplification in the visuals")
    c.add_argument("--video-fps", type=float, default=12)
    c.add_argument("--lpips", action="store_true", help="needs `pip install lpips`")
    c.add_argument("--noise", default=None, metavar="RUN",
                   help="a second run of A's side: its difference to A is the noise floor")

    args = p.parse_args()
    if args.cmd == "run":
        args.clip = os.path.abspath(args.clip)
        args.out = os.path.abspath(args.out)
        if not os.path.isabs(args.config):
            args.config = os.path.join(os.path.abspath(args.repo), args.config)
        cmd_run(args)
    else:
        cmd_compare(args)


if __name__ == "__main__":
    main()
