"""
FastAPI inference server for self-hosted avatar lip-sync.

Design decisions that matter on a single-GPU pod:
  * ONE worker thread drains a queue. Two diffusion jobs on one GPU just thrash
    VRAM and finish slower than running them back-to-back, so we serialise.
  * The model scripts are invoked per-job via the adapters. (If you later port
    to an in-process resident model, swap the adapter call for a function call
    and keep everything else.)
  * Jobs are tracked in a dict + on disk, so you can poll status and fetch the
    finished mp4. State is in-memory; a pod restart loses the queue, which is
    fine for this workload.

Endpoints:
  POST /generate     image, audio, prompt, model, resolution  -> {job_id}
  GET  /status/{id}  -> {status, error?, duration_s?}
  GET  /result/{id}  -> the mp4 file
  GET  /healthz      -> liveness + which models are installed
"""
import threading
import queue
import time
import shutil
import traceback
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from . import config
from .utils import audio_duration_seconds, new_job_id, prep_image
from .adapters import multitalk_adapter, hunyuan_adapter

app = FastAPI(title="Avatar Lip-Sync Server", version="1.0")


class Status(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    model: str
    resolution: str
    prompt: str
    image_path: str
    audio_path: str
    status: Status = Status.QUEUED
    error: str | None = None
    output_path: str | None = None
    queued_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def public(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        if self.started_at and self.finished_at:
            d["duration_s"] = round(self.finished_at - self.started_at, 1)
        # don't leak absolute server paths
        d.pop("image_path", None)
        d.pop("audio_path", None)
        d.pop("output_path", None)
        d["has_result"] = self.output_path is not None
        return d


JOBS: dict[str, Job] = {}
WORK_Q: "queue.Queue[str]" = queue.Queue()

ADAPTERS = {
    "multitalk": multitalk_adapter,
    "hunyuan": hunyuan_adapter,
}


def _process(job: Job) -> None:
    """Run one job to completion. Executed only by the single worker thread."""
    job.status = Status.RUNNING
    job.started_at = time.time()
    job_dir = config.JOB_DIR / job.id
    job_dir.mkdir(parents=True, exist_ok=True)

    adapter = ADAPTERS[job.model]
    produced = adapter.generate(
        job_dir=job_dir,
        image_path=Path(job.image_path),
        audio_path=Path(job.audio_path),
        prompt=job.prompt,
        resolution=job.resolution,
    )

    # Copy the result somewhere stable and predictable.
    final = config.OUTPUT_DIR / f"{job.id}.mp4"
    shutil.copy(produced, final)
    job.output_path = str(final)
    job.status = Status.DONE


def _worker() -> None:
    while True:
        job_id = WORK_Q.get()
        job = JOBS.get(job_id)
        if job is None:
            WORK_Q.task_done()
            continue
        try:
            _process(job)
        except Exception as exc:  # noqa: BLE001 - we want every failure recorded
            job.status = Status.FAILED
            job.error = f"{exc}\n{traceback.format_exc()}"
        finally:
            job.finished_at = time.time()
            WORK_Q.task_done()


# Start exactly one worker. This is deliberate (see module docstring).
threading.Thread(target=_worker, daemon=True).start()


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "ok",
        "models_installed": {
            "multitalk": config.MULTITALK_CKPT.exists() and config.WAN_BASE_CKPT.exists(),
            "hunyuan": config.HUNYUAN_CKPT.exists(),
        },
        "queue_depth": WORK_Q.qsize(),
    }


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    prompt: str = Form("A person speaking naturally to camera, warm and engaged."),
    model: str = Form("multitalk"),
    resolution: str = Form(config.DEFAULT_RESOLUTION),
) -> JSONResponse:
    if model not in ADAPTERS:
        raise HTTPException(400, f"model must be one of {list(ADAPTERS)}")
    if resolution not in ("480", "720"):
        raise HTTPException(400, "resolution must be '480' or '720'")

    job_id = new_job_id()
    job_dir = config.JOB_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Persist uploads
    raw_img = job_dir / f"image_{image.filename}"
    raw_aud = job_dir / f"audio_{audio.filename}"
    raw_img.write_bytes(await image.read())
    raw_aud.write_bytes(await audio.read())

    # Guard against runaway audio length
    dur = audio_duration_seconds(raw_aud)
    if dur > config.MAX_AUDIO_SECONDS:
        raise HTTPException(
            400, f"audio is {dur:.0f}s; max allowed is {config.MAX_AUDIO_SECONDS}s"
        )

    # Sanitise the portrait to the target short-side
    clean_img = prep_image(raw_img, job_dir / "image_clean.jpg",
                           target_short_side=int(resolution))

    job = Job(
        id=job_id, model=model, resolution=resolution, prompt=prompt,
        image_path=str(clean_img), audio_path=str(raw_aud),
    )
    JOBS[job_id] = job
    WORK_Q.put(job_id)
    return JSONResponse({"job_id": job_id, "audio_seconds": round(dur, 1),
                         "queue_position": WORK_Q.qsize()})


@app.get("/status/{job_id}")
def status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job_id")
    return job.public()


@app.get("/result/{job_id}")
def result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job_id")
    if job.status != Status.DONE or not job.output_path:
        raise HTTPException(409, f"job not done (status={job.status.value})")
    return FileResponse(job.output_path, media_type="video/mp4",
                        filename=f"{job_id}.mp4")
