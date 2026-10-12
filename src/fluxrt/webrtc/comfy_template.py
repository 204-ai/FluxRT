"""
Load the ComfyUI API-format workflow that /comfy/edit patches and queues.

The template is exported per machine (``scripts/comfy_workflows/*.json`` is
gitignored), so a fresh checkout has none. Keep the two failure kinds apart so the
route can answer honestly:

- ``TemplateMissing``: no file at the path. The feature is not set up on this
  machine; the route answers 501 with how to export it.
- ``TemplateInvalid``: the file exists but cannot be used (unreadable, not JSON,
  or missing a node the route patches). The route answers 500.

Torch-free so it can be unit-tested on a laptop (tests/webrtc).
"""

from __future__ import annotations

import json
import os

# Node ids /comfy/edit patches in the exported Qwen-Image-Edit 2509 graph.
REQUIRED_NODES = ("78", "111", "3")


class TemplateMissing(Exception):
    """No workflow template at the expected path."""


class TemplateInvalid(Exception):
    """A workflow template exists but cannot be used."""


def load_edit_template(path: str, required_nodes=REQUIRED_NODES) -> dict:
    name = os.path.basename(path)
    if not os.path.isfile(path):
        raise TemplateMissing(
            f"ComfyUI edit workflow not found: {path}. It is exported per machine: "
            f"open the Qwen-Image-Edit 2509 workflow in ComfyUI, use Workflow → "
            f"Export (API), and save it as {name} in {os.path.dirname(path)}."
        )
    try:
        with open(path, "r") as f:
            wf = json.load(f)
    except (OSError, ValueError) as exc:
        raise TemplateInvalid(f"Workflow template unreadable ({path}): {exc}") from exc
    if not isinstance(wf, dict):
        raise TemplateInvalid(f"Workflow template is not an API-format graph: {path}")
    for node_id in required_nodes:
        node = wf.get(node_id)
        if not isinstance(node, dict) or "inputs" not in node:
            raise TemplateInvalid(f"Workflow template missing node {node_id} — re-export {name}")
    return wf
