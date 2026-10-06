"""Prompt list for the fluxrt COMP (Prompts page), the realtime client's prompt
player in TouchDesigner form.

A list comes from a JSON file (a list, or {"prompts": [...]}, entries
{prompt, style, tracking, stability, verdict?} - prompt-fluxrt-v2.json,
scripts/saved_prompts.json, ...) or from the server's GET /prompts, and lives in
table_prompts so it is saved with the project. Filter (verdict) and Search narrow
the Prompt menu; Prev / Next / Shuffle / Autoplay walk the narrowed list. Picking
a prompt sets the Prompt par, so it goes to the server like a typed one (Travel
seconds included).
"""

import json
import os
import random
import time

VERDICT_TAG = {"love": "<3", "like": "+", "skip": "-"}
COLUMNS = ["prompt", "verdict", "style", "tracking", "stability"]
_auto = {"last": 0.0}
_list = {"quiet_until": -1, "autoload_tried": False}


def _comp():
    return parent.FluxRT


def _table():
    return op("table_prompts")


def _info(text):
    _comp().par.Listinfo = text


def entries():
    t = _table()
    if t.numRows < 2:
        return []
    return [{name: t[r, name].val for name in COLUMNS} for r in range(1, t.numRows)]


def _parse(data):
    if isinstance(data, dict):
        data = data.get("prompts", [])
    out = []
    for e in data if isinstance(data, list) else []:
        if isinstance(e, str):
            e = {"prompt": e}
        if not isinstance(e, dict):
            continue
        prompt = str(e.get("prompt", "")).strip()
        if prompt:
            out.append([prompt, e.get("verdict") or ""] + [int(e.get(k) or 0) for k in COLUMNS[2:]])
    return out


def _store(rows, source):
    t = _table()
    t.clear()
    t.appendRow(COLUMNS)
    for row in rows:
        t.appendRow(row)
    # kept in the COMP's storage: survives a DAT edit and is saved with the project
    _comp().store("prompt_source", "{} prompts from {}".format(len(rows), source))
    refresh_menu()


def load_file():
    path = _comp().par.Promptfile.eval()
    try:
        with open(path, encoding="utf-8") as f:
            rows = _parse(json.load(f))
    except (OSError, ValueError) as e:
        _info("could not read {}: {}".format(path, e))
        return
    if not rows:
        _info("no prompts in " + os.path.basename(path))
        return
    _store(rows, os.path.basename(path))


def load_server():
    client = op("client").module
    op("webclient_prompts").request(client._base() + "/prompts", "GET", timeout=3000)
    _info("loading from server...")


def on_prompts(status, data):
    if status.get("code") != 200:
        _info("server /prompts: {}".format(status.get("code")))
        return
    try:
        text = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)
        rows = _parse(json.loads(text))
    except ValueError as e:
        _info("server /prompts: " + str(e))
        return
    _store(rows, "server")


def visible():
    """Indices of the entries that pass Filter and Search, in list order."""
    c = _comp()
    verdict = c.par.Filter.eval()
    query = c.par.Search.eval().strip().lower()
    return [
        i for i, e in enumerate(entries())
        if (verdict == "all" or e["verdict"] == verdict)
        and (not query or query in e["prompt"].lower())
    ]


def _label(i, e):
    text = e["prompt"] if len(e["prompt"]) <= 90 else e["prompt"][:89] + "..."
    return "{:>3}  {:2} {}".format(i + 1, VERDICT_TAG.get(e["verdict"], "."), text)


def refresh_menu():
    par = _comp().par.Promptindex
    all_entries = entries()
    pool = visible()
    current = par.eval()
    # A narrowed menu that drops the current pick moves the par to its first
    # entry; that is not a pick, so don't send it (see quiet()).
    _list["quiet_until"] = absTime.frame + 1
    par.menuNames = [str(i) for i in pool] or ["-1"]
    par.menuLabels = [_label(i, all_entries[i]) for i in pool] or ["(no prompts)"]
    if current in par.menuNames:
        par.val = current
    shown = "" if len(pool) == len(all_entries) else ", {} shown".format(len(pool))
    _info(_comp().fetch("prompt_source", "{} prompts".format(len(all_entries)), search=False) + shown)


def quiet():
    return absTime.frame <= _list["quiet_until"]


def apply_index(index):
    all_entries = entries()
    if not 0 <= index < len(all_entries):
        return
    _comp().par.Prompt = all_entries[index]["prompt"]


def _current():
    try:
        return int(_comp().par.Promptindex.eval())
    except ValueError:
        return -1


def _select(index):
    par = _comp().par.Promptindex
    if par.eval() == str(index):
        apply_index(index)  # same menu value: no change callback, so send it here
    else:
        par.val = str(index)


def step(direction):
    pool = visible()
    if not pool:
        return
    cur = _current()
    if cur in pool:
        nxt = pool[(pool.index(cur) + direction) % len(pool)]
    else:
        nxt = pool[0 if direction > 0 else -1]
    _select(nxt)


def shuffle():
    pool = visible()
    if not pool:
        return
    cur = _current()
    choices = [i for i in pool if i != cur] or pool
    _select(random.choice(choices))


def reset_timer():
    _auto["last"] = time.time()


def tick():
    """Called every frame by execute_poll: loads Prompt File once when the list
    is empty (a fresh FluxRT.tox), then runs Autoplay."""
    c = _comp()
    if not _list["autoload_tried"]:
        _list["autoload_tried"] = True
        if _table().numRows < 2 and os.path.isfile(c.par.Promptfile.eval()):
            load_file()
    if not c.par.Autoplay or time.time() - _auto["last"] < float(c.par.Interval):
        return
    _auto["last"] = time.time()
    if c.par.Random:
        shuffle()
    else:
        step(1)

