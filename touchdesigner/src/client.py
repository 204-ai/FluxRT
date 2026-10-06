"""FluxRT server client for the fluxrt COMP.

Control goes to run_webrtc.py over HTTP (POST /prompt, /prompt-travel, /seed,
/steps) and /healthz is polled once a second; both through Web Client DATs, so
nothing blocks the TouchDesigner frame. Frames go over Spout (spoutout_input ->
server --spout-in, server --spout-out -> spoutin_output), not through here.
"""

import json
import time

_state = {"last_ok": 0.0, "quiet_until": -1, "pull": False, "health": None,
          "status_frame": -1, "status": None}

GREEN = (0.08, 0.50, 0.22)
AMBER = (0.72, 0.46, 0.04)
RED = (0.68, 0.10, 0.10)
GREY = (0.30, 0.30, 0.30)


def _comp():
    return parent.FluxRT


def _base():
    return _comp().par.Server.eval().strip().rstrip("/")


def _post(path, payload):
    op("webclient_ctrl").request(
        _base() + path,
        "POST",
        header={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=3000,
    )


def quiet():
    """True while Pull State writes the server's values into the pars, so the
    parameter callbacks don't send them straight back."""
    return absTime.frame <= _state["quiet_until"]


def send_prompt():
    c = _comp()
    prompt = c.par.Prompt.eval().strip()
    if not prompt:
        return
    seconds = float(c.par.Travel)
    if seconds > 0:
        _post("/prompt-travel", {"prompt": prompt, "frames": f"{seconds:g}s", "mode": "slerp"})
    else:
        _post("/prompt", {"prompt": prompt})


def send_seed():
    _post("/seed", {"value": int(_comp().par.Seed)})


def send_steps():
    _post("/steps", {"value": int(_comp().par.Steps)})


def pull_state():
    _state["pull"] = True
    poll()


def poll():
    c = _comp()
    if not c.par.Active:
        return
    op("webclient_health").request(_base() + "/healthz", "GET", timeout=2000)
    if time.time() - _state["last_ok"] > 3 and c.par.Ready:
        c.par.Ready = False
        c.par.Inputsource = "server unreachable"


def _text(data):
    return data.decode("utf-8", "replace") if isinstance(data, (bytes, bytearray)) else str(data)


def on_ctrl(status, data):
    _comp().par.Lastcontrol = f"{status.get('code')} {_text(data)[:120]}"
    if status.get("code") != 200:
        print("FluxRT control:", status, _text(data)[:300])


def _set(par, value):
    if par.eval() != value:
        par.val = value


def on_health(status, data):
    if status.get("code") != 200:
        return
    try:
        h = json.loads(_text(data))
    except ValueError:
        return
    _state["last_ok"] = time.time()
    _state["health"] = h
    c = _comp()
    _set(c.par.Ready, bool(h.get("ready")))
    _set(c.par.Fpsgen, round(float(h.get("fps_pipeline", 0)), 1))
    _set(c.par.Fpsout, round(float(h.get("fps_interpolated", 0)), 1))
    _set(c.par.Procms, round(float(h.get("proc_time_ms", 0)), 1))
    _set(c.par.Inputsource, str(h.get("input_source", "")))
    res = h.get("resolution") or {}
    if res.get("width") and res.get("height"):
        _set(c.par.Inputw, int(res["width"]))
        _set(c.par.Inputh, int(res["height"]))

    spout = h.get("spout") or {}
    table = op("table_status")
    table.clear()
    for key, value in (
        ("ready", h.get("ready")),
        ("fps generated", h.get("fps_pipeline")),
        ("fps out", h.get("fps_interpolated")),
        ("ms per frame", h.get("proc_time_ms")),
        ("input", h.get("input_source")),
        ("spout in", f"{spout.get('in')} {spout.get('in_size')}"),
        ("spout out", spout.get("out")),
        ("prompt", h.get("prompt")),
        ("seed", h.get("seed")),
        ("steps", h.get("steps")),
        ("vram MB", h.get("vram_mb")),
    ):
        table.appendRow([key, value])

    if _state["pull"]:
        _state["pull"] = False
        _state["quiet_until"] = absTime.frame + 2
        c.par.Prompt = h.get("prompt", "")
        c.par.Seed = int(h.get("seed", c.par.Seed))
        c.par.Steps = int(h.get("steps", c.par.Steps))


LIVE, WARN, FAULT, PAUSED = "●", "▲", "✕", "⏸"   # ● ▲ ✕ ⏸
GEN, INTERP, UPSCALE = "⚡", "⇉", "⤢"                  # ⚡ ⇉ ⤢


def _fps(v):
    return "{:.0f}".format(v) if v >= 10 else "{:.1f}".format(v)


def status():
    """(state, glyph, colour, hud) for the viewer overlay: a one-line pill,
    `● ⚡53  ⇉×2 107  ⤢×2 1152×640` = state, generated fps, RIFE factor and
    output fps, upscale factor and output size. The state in words goes to the
    Status page (State). Computed once per frame: the text and the three colour
    channels all ask for it."""
    if _state["status_frame"] == absTime.frame:
        return _state["status"]
    c = _comp()
    h = _state["health"] or {}
    spout = h.get("spout") or {}
    online = _state["health"] is not None and time.time() - _state["last_ok"] < 3
    receiving = op("info_spoutin")["sender_fps"].eval() > 0.5 and op("spoutin_output").width > 128

    if not c.par.Active:
        state, glyph, colour = "paused", PAUSED, GREY
    elif not online:
        state, glyph, colour = "server offline: " + _base(), FAULT, RED
    elif not h.get("ready"):
        state, glyph, colour = "server warming up", WARN, AMBER
    elif h.get("input_source") == "peer":
        state, glyph, colour = "a browser has the input", WARN, AMBER
    elif not spout.get("in"):
        state, glyph, colour = "server runs without --spout-in", WARN, AMBER
    elif not spout.get("in_size"):
        state, glyph, colour = "server not receiving " + c.par.Inputsender.eval(), WARN, AMBER
    elif not spout.get("out"):
        state, glyph, colour = "server runs without --spout-out", WARN, AMBER
    elif not receiving:
        state, glyph, colour = "no frames from " + c.par.Outputsender.eval(), FAULT, RED
    else:
        state, glyph, colour = "live", LIVE, GREEN

    hud = glyph
    if online and h.get("ready"):
        gen = float(h.get("fps_pipeline", 0))
        out_fps = float(h.get("fps_interpolated", 0))
        interp = h.get("interpolation") or (round(out_fps / gen) if gen > 0 else 1)
        res_in = h.get("resolution") or {}
        res_out = h.get("out_resolution") or {"width": op("spoutin_output").width,
                                              "height": op("spoutin_output").height}
        scale = res_out["width"] / res_in["width"] if res_in.get("width") else 1
        hud += "  {}{}  {}×{} {}  {}×{:g} {}×{}".format(
            GEN, _fps(gen), INTERP, interp, _fps(out_fps),
            UPSCALE, round(scale, 2), res_out["width"], res_out["height"])
    _state["status_frame"] = absTime.frame
    _state["status"] = (state, glyph, colour, hud)
    _set(c.par.State, state)
    return _state["status"]


def hud_text():
    return status()[3]


def status_colour(i):
    return status()[2][i]
