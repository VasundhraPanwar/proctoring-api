"""
api.py — FastAPI service with file-based job persistence
Jobs survive Railway restarts by storing to disk as JSON files
"""

import os
import uuid
import json
import shutil
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

UPLOAD_DIR = Path("./uploads")
JOBS_DIR   = Path("./jobs")
UPLOAD_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Proctoring API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class JobStatus(BaseModel):
    job_id:       str
    status:       str
    created_at:   str
    completed_at: Optional[str] = None
    result:       Optional[dict] = None
    error:        Optional[str]  = None


# ── Persistent job store (files on disk) ───────────────────────────────────────
def save_job(job: dict):
    path = JOBS_DIR / f"{job['job_id']}.json"
    with open(path, "w") as f:
        json.dump(job, f)

def load_job(job_id: str) -> Optional[dict]:
    path = JOBS_DIR / f"{job_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def update_job(job_id: str, **kwargs):
    job = load_job(job_id)
    if job:
        job.update(kwargs)
        save_job(job)


# ── Detection runner ───────────────────────────────────────────────────────────
async def run_job(job_id: str, video_path: Path, student_id: str):
    update_job(job_id, status="processing")
    try:
        loop = asyncio.get_event_loop()

        def _detect():
            from detector import InterviewCheatingDetector
            d = InterviewCheatingDetector(str(video_path), student_id=student_id)
            return d.analyze().to_dict()

        result = await loop.run_in_executor(None, _detect)
        update_job(job_id,
            status       = "done",
            completed_at = datetime.utcnow().isoformat(),
            result       = result,
        )
    except Exception as e:
        update_job(job_id,
            status       = "failed",
            completed_at = datetime.utcnow().isoformat(),
            error        = str(e),
        )
    finally:
        try: video_path.unlink(missing_ok=True)
        except: pass


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/analyze", response_model=JobStatus, status_code=202)
async def analyze(
    background_tasks: BackgroundTasks,
    video:      UploadFile = File(...),
    student_id: str = "",
):
    ext = Path(video.filename).suffix.lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    job_id     = str(uuid.uuid4())
    save_path  = UPLOAD_DIR / f"{job_id}{ext}"
    student_id = student_id or Path(video.filename).stem

    with open(save_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    job = {
        "job_id":       job_id,
        "status":       "queued",
        "created_at":   datetime.utcnow().isoformat(),
        "completed_at": None,
        "result":       None,
        "error":        None,
    }
    save_job(job)

    background_tasks.add_task(run_job, job_id, save_path, student_id)
    return JobStatus(**job)


@app.get("/status/{job_id}", response_model=JobStatus)
def status(job_id: str):
    job = load_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JobStatus(**job)


@app.get("/jobs")
def list_jobs():
    jobs = []
    for f in JOBS_DIR.glob("*.json"):
        try:
            with open(f) as jf:
                jobs.append(json.load(jf))
        except:
            pass
    return sorted(jobs, key=lambda j: j.get("created_at",""), reverse=True)
