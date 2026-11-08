import json
import queue
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from shared.transformer_core import (
    ExperimentConfig,
    generate_series,
    run_experiment,
)


app = FastAPI(
    title="时序 Transformer API",
    description=(
        "使用 FastAPI 提供时序数据、"
        "Transformer 训练和 SSE 实时进度。"
    ),
    version="1.0.0",
)


CURRENT_DIRECTORY = Path(__file__).parent
STATIC_DIRECTORY = CURRENT_DIRECTORY / "static"

jobs = {}
jobs_lock = threading.Lock()


class TrainRequest(BaseModel):
    series_length: int = Field(
        default=600,
        ge=300,
        le=1200,
    )

    window_size: int = Field(
        default=24,
        ge=12,
        le=60,
    )

    epochs: int = Field(
        default=10,
        ge=5,
        le=30,
    )

    seed: int = Field(
        default=42,
        ge=0,
        le=9999,
    )

    use_mps: bool = False


@app.get("/")
def index():
    return FileResponse(
        STATIC_DIRECTORY / "index.html"
    )


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "framework": "FastAPI",
    }


@app.get("/api/preview")
def preview(
    series_length: int = Query(
        default=600,
        ge=300,
        le=1200,
    ),
    seed: int = Query(
        default=42,
        ge=0,
        le=9999,
    ),
):
    _, prices = generate_series(
        length=series_length,
        seed=seed,
    )

    series = [
        {
            "time": int(index),
            "value": float(value),
        }
        for index, value in enumerate(prices)
    ]

    return {
        "series": series,
        "series_length": series_length,
        "seed": seed,
    }


def run_training_job(
    job_id: str,
    train_request: TrainRequest,
):
    event_queue = jobs[job_id]["event_queue"]

    config = ExperimentConfig(
        series_length=train_request.series_length,
        window_size=train_request.window_size,
        epochs=train_request.epochs,
        seed=train_request.seed,
        use_mps=train_request.use_mps,
        epoch_delay=0.1,
    )

    def send_progress(progress):
        event_queue.put(
            {
                "event": "progress",
                "data": progress,
            }
        )

    try:
        output = run_experiment(
            config=config,
            progress_callback=send_progress,
        )

        with jobs_lock:
            jobs[job_id]["status"] = "completed"

        event_queue.put(
            {
                "event": "completed",
                "data": output,
            }
        )

    except Exception as error:
        with jobs_lock:
            jobs[job_id]["status"] = "failed"

        event_queue.put(
            {
                "event": "failed",
                "data": {
                    "message": str(error),
                },
            }
        )


@app.post(
    "/api/train",
    status_code=202,
)
def start_training(
    train_request: TrainRequest,
):
    job_id = str(uuid.uuid4())
    event_queue = queue.Queue()

    with jobs_lock:
        jobs[job_id] = {
            "status": "running",
            "event_queue": event_queue,
        }

    worker = threading.Thread(
        target=run_training_job,
        args=(job_id, train_request),
        daemon=True,
    )

    worker.start()

    return {
        "job_id": job_id,
        "status": "running",
        "events_url": f"/api/events/{job_id}",
    }


def format_sse_event(
    event_name: str,
    data,
):
    json_data = json.dumps(
        data,
        ensure_ascii=False,
    )

    return (
        f"event: {event_name}\n"
        f"data: {json_data}\n\n"
    )


@app.get("/api/events/{job_id}")
def training_events(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="找不到这个训练任务",
        )

    event_queue = job["event_queue"]

    def event_stream():
        # 先告诉浏览器连接已经建立
        yield format_sse_event(
            "connected",
            {
                "job_id": job_id,
                "message": "SSE连接已经建立",
            },
        )

        while True:
            event = event_queue.get()

            yield format_sse_event(
                event["event"],
                event["data"],
            )

            if event["event"] in {
                "completed",
                "failed",
            }:
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )