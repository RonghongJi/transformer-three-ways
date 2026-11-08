import threading
import uuid

from flask import Flask, jsonify, render_template, request

from shared.transformer_core import (
    ExperimentConfig,
    generate_series,
    run_experiment,
)


app = Flask(__name__)

# 临时把任务状态保存在内存中。
# 正式系统通常会使用数据库或任务队列。
jobs = {}
jobs_lock = threading.Lock()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "framework": "Flask",
        }
    )


@app.get("/api/preview")
def preview():
    series_length = request.args.get(
        "series_length",
        default=600,
        type=int,
    )

    seed = request.args.get(
        "seed",
        default=42,
        type=int,
    )

    series_length = max(
        300,
        min(series_length, 1200),
    )

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

    return jsonify(
        {
            "series": series,
            "series_length": series_length,
            "seed": seed,
        }
    )


def run_training_job(
    job_id: str,
    config: ExperimentConfig,
):
    def update_progress(progress):
        with jobs_lock:
            jobs[job_id]["progress"] = progress[
                "progress"
            ]
            jobs[job_id]["epoch"] = progress[
                "epoch"
            ]
            jobs[job_id]["epochs"] = progress[
                "epochs"
            ]
            jobs[job_id]["loss"] = progress[
                "loss"
            ]
            jobs[job_id]["accuracy"] = progress[
                "accuracy"
            ]
            jobs[job_id]["history"].append(
                progress
            )

    try:
        output = run_experiment(
            config=config,
            progress_callback=update_progress,
        )

        with jobs_lock:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["output"] = output

    except Exception as error:
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(error)


@app.post("/api/train")
def start_training():
    data = request.get_json(
        silent=True,
    ) or {}

    series_length = int(
        data.get("series_length", 600)
    )

    window_size = int(
        data.get("window_size", 24)
    )

    epochs = int(
        data.get("epochs", 10)
    )

    seed = int(
        data.get("seed", 42)
    )

    use_mps = bool(
        data.get("use_mps", False)
    )

    # 限制参数，避免用户提交过大的训练任务
    series_length = max(
        300,
        min(series_length, 1200),
    )

    window_size = max(
        12,
        min(window_size, 60),
    )

    epochs = max(
        5,
        min(epochs, 30),
    )

    config = ExperimentConfig(
        series_length=series_length,
        window_size=window_size,
        epochs=epochs,
        seed=seed,
        use_mps=use_mps,
        epoch_delay=0.1,
    )

    job_id = str(uuid.uuid4())

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "progress": 0,
            "epoch": 0,
            "epochs": epochs,
            "loss": None,
            "accuracy": None,
            "history": [],
            "output": None,
            "error": None,
        }

    worker = threading.Thread(
        target=run_training_job,
        args=(job_id, config),
        daemon=True,
    )

    worker.start()

    return (
        jsonify(
            {
                "job_id": job_id,
                "status": "running",
            }
        ),
        202,
    )


@app.get("/api/status/<job_id>")
def get_training_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            return (
                jsonify(
                    {
                        "error": "找不到这个训练任务",
                    }
                ),
                404,
            )

        # 创建副本，避免序列化过程中被后台线程修改
        response = {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job["progress"],
            "epoch": job["epoch"],
            "epochs": job["epochs"],
            "loss": job["loss"],
            "accuracy": job["accuracy"],
            "history": list(job["history"]),
            "output": job["output"],
            "error": job["error"],
        }

    return jsonify(response)


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )