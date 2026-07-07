"""
Celery task for FLUX.1-dev Q8 GGUF image generation.

Key design decisions
--------------------
* The FLUX pipeline is loaded **once** at module level (worker boot) to prevent
  repeated VRAM allocation / OOM across tasks.
* Image bytes arrive hex-encoded (JSON-safe) and are decoded back before use.
* Generated images are saved to ``/data/generated/<task_id>.png`` — the shared
  Docker volume makes these accessible from the API container.
* Task status is written directly to Postgres via SQLAlchemy so the FastAPI
  ``GET /status/{task_id}`` endpoint can serve it without touching Redis.
"""

import io
import os
import uuid
import traceback

import torch
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from worker.celery_app import app

# ---------------------------------------------------------------------------
# Database session (worker-side) — mirrors app/database.py but uses its own
# engine to avoid import-time side effects with FastAPI's lifespan.
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "postgresql://kortex:kortex_secret@db:5432/kortex_tasks",
)
_engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=2)
_SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

# ---------------------------------------------------------------------------
# Global ML pipeline — loaded exactly ONCE when the worker process boots.
# ---------------------------------------------------------------------------
OUTPUT_DIR = "/data/generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FLUX_GGUF_URL = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q8_0.gguf"

_pipe = None  # Lazy-initialised on first task or worker signal


def _load_pipeline():
    """Load the FLUX.1-dev Q8 GGUF pipeline with CPU offloading."""
    global _pipe
    if _pipe is not None:
        return _pipe

    print(f"[Worker] Loading FLUX.1-dev Q8 GGUF from {FLUX_GGUF_URL} …")

    from diffusers import FluxPipeline

    _pipe = FluxPipeline.from_single_file(
        FLUX_GGUF_URL,
        torch_dtype=torch.bfloat16,
    )
    _pipe.enable_model_cpu_offload()

    print("[Worker] FLUX.1-dev Q8 GGUF pipeline ready (bfloat16 + CPU offload)")
    return _pipe


# Eagerly load on import (= worker boot) so the first task isn't slow.
try:
    _load_pipeline()
except Exception as exc:
    # Don't crash the worker if GPU isn't available at import time
    # (e.g. during unit tests or API-only containers).
    print(f"[Worker] Deferred pipeline load — {exc}")


# ---------------------------------------------------------------------------
# Helper: update Task row in Postgres
# ---------------------------------------------------------------------------
def _update_task(task_id: str, status: str, result_url: str | None = None):
    """Write status + result_url to the tasks table."""
    # Import model here to avoid circular import at module level
    from app.models import Task

    session = _SessionLocal()
    try:
        task_uuid = uuid.UUID(task_id)
        task_row = session.query(Task).filter(Task.id == task_uuid).first()
        if task_row:
            task_row.status = status
            if result_url is not None:
                task_row.result_url = result_url
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------
@app.task(name="worker.ml_tasks.run_generation", bind=True, max_retries=1)
def run_generation(
    self,
    task_id: str,
    image_bytes: str,
    mask_bytes: str,
    prompt: str,
    vibe_strength: float = 0.0,
):
    """
    Execute a FLUX.1-dev Q8 GGUF generation for the given prompt and input images.

    Parameters
    ----------
    task_id : str
        UUID of the Task row in Postgres.
    image_bytes : str
        Hex-encoded PNG bytes of the source image.
    mask_bytes : str
        Hex-encoded PNG bytes of the mask image.
    prompt : str
        Text prompt describing the desired generation.
    vibe_strength : float
        Placeholder for future vibe-matching (currently unused by FLUX).
    """
    pipe = _load_pipeline()

    _update_task(task_id, status="PROCESSING")

    try:
        # ----- Decode inputs -----
        pil_image = Image.open(io.BytesIO(bytes.fromhex(image_bytes))).convert("RGB")
        pil_mask = Image.open(io.BytesIO(bytes.fromhex(mask_bytes))).convert("RGB")

        # Resize to model-friendly resolution
        work_size = (1024, 1024)
        pil_image = pil_image.resize(work_size, Image.Resampling.LANCZOS)
        pil_mask = pil_mask.resize(work_size, Image.Resampling.NEAREST)

        # ----- Run inference -----
        # FLUX.1-dev uses classifier-free guidance at 3.5 and 25 denoising
        # steps for high-quality output from the Q8 GGUF weights.
        result = pipe(
            prompt=prompt,
            num_inference_steps=25,
            guidance_scale=3.5,
            height=work_size[1],
            width=work_size[0],
        ).images[0]

        # ----- Save output -----
        output_path = os.path.join(OUTPUT_DIR, f"{task_id}.png")
        result.save(output_path, format="PNG")

        _update_task(task_id, status="SUCCESS", result_url=output_path)

        return {"task_id": task_id, "status": "SUCCESS", "result_url": output_path}

    except Exception as exc:
        traceback.print_exc()
        _update_task(task_id, status="FAILURE", result_url=None)
        raise self.retry(exc=exc, countdown=5)
