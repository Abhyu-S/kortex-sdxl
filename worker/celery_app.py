"""
Celery application instance for the Kortex worker.

Connects to the Redis broker defined in docker-compose.yml and
auto-discovers task modules in the `worker` package.
"""

import os
from celery import Celery

CELERY_BROKER_URL: str = os.environ.get(
    "CELERY_BROKER_URL",
    "redis://redis:6379/0",
)
CELERY_RESULT_BACKEND: str = os.environ.get(
    "CELERY_RESULT_BACKEND",
    "redis://redis:6379/0",
)

app = Celery(
    "kortex_worker",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["worker.ml_tasks"],
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Prevent the worker from pre-fetching tasks; one GPU task at a time.
    worker_prefetch_multiplier=1,
    # Acknowledge only after the task completes (no lost GPU work).
    task_acks_late=True,
    # Reject tasks back to the queue on worker shutdown (graceful restart).
    task_reject_on_worker_lost=True,
)
