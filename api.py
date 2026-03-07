"""
api.py — FastAPI service for cheating detection
Exposes /analyze (upload video) and /status/{job_id} for n8n polling
"""

import os
import uuid
import shutil
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

jobs: dict[str, dict] = {}

app = FastAPI(title="Proctoring API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class JobStatus(BaseModel):
    job_id:       str
    status:       str
    created_at:   str
    completed_at: Optional[str] = None
    result:       Optional[dict] = None
    error:        Optional[str]  = None


async def run_job(job_id: str, video_path: Path, student_id: str):
    jobs[job_id]["status"] = "processing"
    try:
        # Run detector in thread pool so it doesn't block the event loop
        loop = asyncio.get_event_loop()

        def _detect():
            from detector import CheatingDetector
            d = CheatingDetector(str(video_path), student_id=student_id)
            return d.analyze().to_dict()

        result = await loop.run_in_executor(None, _detect)

        jobs[job_id].update({
            "status":       "done",
            "completed_at": datetime.utcnow().isoformat(),
            "result":       result,
        })
    except Exception as e:
        jobs[job_id].update({
            "status":       "failed",
            "completed_at": datetime.utcnow().isoformat(),
            "error":        str(e),
        })
    finally:
        try: video_path.unlink(missing_ok=True)
        except: pass


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/analyze", response_model=JobStatus, status_code=202)
async def analyze(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
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

    jobs[job_id] = {
        "job_id":       job_id,
        "status":       "queued",
        "created_at":   datetime.utcnow().isoformat(),
        "completed_at": None,
        "result":       None,
        "error":        None,
    }

    background_tasks.add_task(run_job, job_id, save_path, student_id)
    return JobStatus(**jobs[job_id])


@app.get("/status/{job_id}", response_model=JobStatus)
def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return JobStatus(**jobs[job_id])


@app.get("/jobs")
def list_jobs():
    return list(jobs.values())
