"""FastAPI routes for the offline batch render, as a router factory so they can be
mounted in run_webrtc.py AND tested standalone against a stub job manager (no CUDA,
no torch). The manager + enabled-flag + default-prompt are injected."""

from __future__ import annotations

import os
from typing import Callable, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

# Cap the uploaded video so a single request can't exhaust host RAM (the whole file
# is buffered, then decoded). Override with FLUXRT_BATCH_MAX_MB.
_MAX_UPLOAD_BYTES = int(os.environ.get("FLUXRT_BATCH_MAX_MB", "256")) * 1024 * 1024


def make_batch_router(
    get_manager: Callable[[], object],
    is_enabled: Callable[[], bool],
    default_prompt: Callable[[], str] = lambda: "",
) -> APIRouter:
    router = APIRouter()

    def _require() -> None:
        if not is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Batch render not enabled. Set FLUXRT_BATCH_RENDER=1 on a host with spare GPU/VRAM (a second full model will not co-fit with the live stream on 24 GB).",
            )

    @router.post("/batch/jobs")
    async def submit(
        video: UploadFile = File(...),
        prompt: str = Form(""),
        seed: int = Form(...),
        steps: int = Form(...),
        fps: Optional[float] = Form(None),
        interp: int = Form(0),
    ):
        """Start an offline render of an uploaded video. interp=0 → 1:1; k>0 → RIFE
        interpolation (2**k frames per input). 202 + job status."""
        _require()
        size = getattr(video, "size", None)  # set by Starlette from the multipart part
        if size is not None and size > _MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"video exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
        data = await video.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty video upload")
        if len(data) > _MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"video exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
        try:
            job = get_manager().submit(
                data,
                prompt=prompt or default_prompt(),
                seed=int(seed),
                steps=int(steps),
                fps=fps,
                interp=int(interp),
            )
        except RuntimeError as exc:  # another job already running
            raise HTTPException(status_code=409, detail=str(exc))
        return JSONResponse(job.status(), status_code=202)

    @router.get("/batch/jobs/{job_id}")
    async def status(job_id: str):
        _require()
        job = get_manager().get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="no such job")
        return job.status()

    @router.get("/batch/jobs/{job_id}/result")
    async def result(job_id: str):
        _require()
        job = get_manager().get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="no such job")
        if job.state != "done" or not job.out_path or not os.path.exists(job.out_path):
            raise HTTPException(status_code=409, detail=f"job not done (state={job.state})")
        return FileResponse(job.out_path, media_type="video/mp4", filename=f"fluxrt-render-{job.id}.mp4")

    @router.delete("/batch/jobs/{job_id}")
    async def cancel(job_id: str):
        _require()
        if not get_manager().cancel(job_id):
            raise HTTPException(status_code=404, detail="no such job")
        return {"ok": True}

    return router
