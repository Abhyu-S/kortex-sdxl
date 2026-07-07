"""
FastAPI application for the Kortex async image-generation engine.

Endpoints:
  POST /generate        – accepts the same multipart payload the mobile app already sends
                          (image, mask, prompt, vibe_strength) and enqueues a Celery task.
  GET  /status/{task_id} – polls the Postgres-backed task state.
  GET  /health           – liveness / GPU probe (backward-compatible).
  GET  /                 – root heartbeat.
"""

import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import engine, Base, get_db
from app.models import Task

# ---------------------------------------------------------------------------
# Lazy import of the Celery task — avoids pulling torch into the API process.
# ---------------------------------------------------------------------------
from worker.celery_app import app as celery_app  # noqa: F401  (registers backend)


# ---------------------------------------------------------------------------
# Lifespan: auto-create DB tables on startup (replaces Alembic)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Create all ORM tables on startup; nothing to tear down."""
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Kortex Generation Engine",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class TaskCreatedResponse(BaseModel):
    task_id: str
    status: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result_url: Optional[str] = None
    created_at: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    """Root heartbeat — kept for backward compatibility."""
    return {"message": "Kortex Generation Engine is running. See /docs for API reference."}


@app.get("/health")
def health_check():
    """Liveness probe matching the original server.py contract."""
    import torch
    return {"status": "running", "gpu": torch.cuda.is_available()}


@app.post("/generate", response_model=TaskCreatedResponse, status_code=202)
async def generate(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...),
    vibe_strength: float = Form(0.0),
    db: Session = Depends(get_db),
):
    """
    Accept the **exact same multipart payload** the mobile app currently sends
    to ``/smart-fill`` and ``/generative-fill``, then push the work into the
    Celery queue for async GPU execution.

    Returns immediately with a ``task_id`` the client can poll via
    ``GET /status/{task_id}``.
    """
    # 1. Read raw bytes from the uploaded files
    image_bytes: bytes = await image.read()
    mask_bytes: bytes = await mask.read()

    # 2. Persist a PENDING task row in Postgres
    task_row = Task(status="PENDING")
    db.add(task_row)
    db.commit()
    db.refresh(task_row)

    task_id_str = str(task_row.id)

    # 3. Enqueue the Celery task (import here to keep module-level light)
    from worker.ml_tasks import run_generation

    run_generation.delay(
        task_id=task_id_str,
        image_bytes=image_bytes.hex(),   # Celery JSON-serialises; hex is safe
        mask_bytes=mask_bytes.hex(),
        prompt=prompt,
        vibe_strength=vibe_strength,
    )

    return TaskCreatedResponse(task_id=task_id_str, status="PENDING")


@app.get("/status/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str, db: Session = Depends(get_db)):
    """Poll the current state of a generation task."""
    try:
        task_uuid = uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task_id format.")

    task_row = db.query(Task).filter(Task.id == task_uuid).first()
    if task_row is None:
        raise HTTPException(status_code=404, detail="Task not found.")

    return TaskStatusResponse(
        task_id=str(task_row.id),
        status=task_row.status,
        result_url=task_row.result_url,
        created_at=task_row.created_at.isoformat(),
    )
