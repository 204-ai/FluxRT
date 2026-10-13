"""Unit tests for the /comfy/edit workflow-template loader (T37 / B4)."""

import json

import pytest

from fluxrt.webrtc.comfy_template import (
    REQUIRED_NODES,
    TemplateInvalid,
    TemplateMissing,
    load_edit_template,
)


def _graph(nodes=REQUIRED_NODES):
    return {n: {"class_type": "X", "inputs": {}} for n in nodes}


def test_missing_template_raises_missing_with_export_hint(tmp_path):
    path = tmp_path / "comfy_workflows" / "qwen_edit_2509.api.json"
    with pytest.raises(TemplateMissing) as exc:
        load_edit_template(str(path))
    msg = str(exc.value)
    assert str(path) in msg
    assert "Export (API)" in msg
    assert "qwen_edit_2509.api.json" in msg


def test_valid_template_loads(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps(_graph()))
    wf = load_edit_template(str(path))
    assert set(REQUIRED_NODES) <= set(wf)


def test_bad_json_is_invalid_not_missing(tmp_path):
    path = tmp_path / "t.json"
    path.write_text("{not json")
    with pytest.raises(TemplateInvalid):
        load_edit_template(str(path))


def test_non_dict_graph_is_invalid(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(TemplateInvalid):
        load_edit_template(str(path))


@pytest.mark.parametrize("missing", REQUIRED_NODES)
def test_missing_patched_node_is_invalid(tmp_path, missing):
    path = tmp_path / "t.json"
    path.write_text(json.dumps(_graph([n for n in REQUIRED_NODES if n != missing])))
    with pytest.raises(TemplateInvalid) as exc:
        load_edit_template(str(path))
    assert f"node {missing}" in str(exc.value)


def test_node_without_inputs_is_invalid(tmp_path):
    graph = _graph()
    graph[REQUIRED_NODES[0]] = {"class_type": "X"}  # no "inputs"
    path = tmp_path / "t.json"
    path.write_text(json.dumps(graph))
    with pytest.raises(TemplateInvalid):
        load_edit_template(str(path))


def test_missing_is_not_a_subclass_of_invalid():
    # the route maps them to different status codes (501 vs 500)
    assert not issubclass(TemplateMissing, TemplateInvalid)
    assert not issubclass(TemplateInvalid, TemplateMissing)
