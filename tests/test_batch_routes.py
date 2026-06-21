"""GPU-free tests for the batch FastAPI routes (batch_routes.make_batch_router)
wired to a real BatchJobManager + a STUB processor. No CUDA / torch."""

import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

import batch_routes
from batch_render import BatchJobManager
from batch_routes import make_batch_router
from test_batch_render import StubProcessor, make_mp4_bytes, count_frames, wait_until


def _client(enabled=True, per_frame_sleep=0.0):
    stubs = []

    def make(cfg):
        s = StubProcessor(cfg, per_frame_sleep=per_frame_sleep)
        stubs.append(s)
        return s

    mgr = BatchJobManager(base_config={"resolution": {"height": 48, "width": 64}}, make_processor=make)
    app = FastAPI()
    app.include_router(make_batch_router(lambda: mgr, lambda: enabled, lambda: "default"))
    return TestClient(app), mgr, stubs


def _post(client, mp4, seed=5, steps=2, prompt=""):
    return client.post(
        "/batch/jobs",
        data={"seed": seed, "steps": steps, "prompt": prompt},
        files={"video": ("in.mp4", mp4, "video/mp4")},
    )


def test_disabled_returns_503():
    client, _, _ = _client(enabled=False)
    r = _post(client, make_mp4_bytes(n_frames=3))
    assert r.status_code == 503


def test_full_job_flow_post_poll_result(tmp_path):
    client, mgr, stubs = _client()
    r = _post(client, make_mp4_bytes(n_frames=5, fps=10), prompt="cat")
    assert r.status_code == 202
    job_id = r.json()["id"]

    # poll status until done
    deadline = time.time() + 10
    state = None
    while time.time() < deadline:
        s = client.get(f"/batch/jobs/{job_id}")
        assert s.status_code == 200
        state = s.json()["state"]
        if state in ("done", "error"):
            break
        time.sleep(0.05)
    assert state == "done"
    assert stubs[0].prompt == "cat"  # default not used when prompt provided

    res = client.get(f"/batch/jobs/{job_id}/result")
    assert res.status_code == 200
    assert res.headers["content-type"] == "video/mp4"
    out = tmp_path / "out.mp4"
    out.write_bytes(res.content)
    assert count_frames(str(out)) == 5  # 1:1


def test_default_prompt_used_when_blank():
    client, mgr, stubs = _client()
    r = _post(client, make_mp4_bytes(n_frames=3), prompt="")
    assert r.status_code == 202
    job_id = r.json()["id"]
    deadline = time.time() + 10
    while time.time() < deadline and mgr.get(job_id).state not in ("done", "error"):
        time.sleep(0.05)
    assert stubs[0].prompt == "default"


def test_unknown_job_404():
    client, _, _ = _client()
    assert client.get("/batch/jobs/nope").status_code == 404
    assert client.delete("/batch/jobs/nope").status_code == 404


def test_result_before_done_is_409():
    client, mgr, stubs = _client(per_frame_sleep=0.05)
    r = _post(client, make_mp4_bytes(n_frames=30))
    job_id = r.json()["id"]
    # running → result not ready (bounded wait; fail loudly if it never gets there)
    assert wait_until(lambda: mgr.get(job_id).state in ("running", "done", "error"))
    assert mgr.get(job_id).state == "running", mgr.get(job_id).error
    assert client.get(f"/batch/jobs/{job_id}/result").status_code == 409
    assert client.delete(f"/batch/jobs/{job_id}").json() == {"ok": True}


def test_second_job_while_running_409():
    client, mgr, stubs = _client(per_frame_sleep=0.05)
    r1 = _post(client, make_mp4_bytes(n_frames=30))
    jid = r1.json()["id"]
    assert wait_until(lambda: mgr.get(jid).state in ("running", "done", "error"))
    assert mgr.get(jid).state == "running", mgr.get(jid).error
    r2 = _post(client, make_mp4_bytes(n_frames=3))
    assert r2.status_code == 409
    client.delete(f"/batch/jobs/{jid}")


def test_oversize_upload_413(monkeypatch):
    monkeypatch.setattr(batch_routes, "_MAX_UPLOAD_BYTES", 1024)  # 1 KB cap
    client, _, _ = _client()
    r = _post(client, make_mp4_bytes(n_frames=20))  # > 1 KB
    assert r.status_code == 413
