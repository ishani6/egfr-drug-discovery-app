"""
Module 10 — Experiment Tracking
Lightweight run logger. No MLflow dependency — plain JSON/CSV.
"""

import json, os, time, uuid
import pandas as pd

LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "logs")
RUN_FILE = os.path.join(LOG_DIR, "runs.json")

os.makedirs(LOG_DIR, exist_ok=True)


def _load_runs() -> list:
    if os.path.exists(RUN_FILE):
        with open(RUN_FILE) as f:
            return json.load(f)
    return []


def _save_runs(runs: list):
    with open(RUN_FILE, "w") as f:
        json.dump(runs, f, indent=2)


def log_run(
    stage: str,
    model_name: str = "",
    dataset_version: str = "v1",
    metrics: dict = None,
    params: dict = None,
    tags: list = None,
) -> str:
    run_id = f"run-{str(uuid.uuid4())[:6].upper()}"
    entry = {
        "run_id":           run_id,
        "timestamp":        time.strftime("%Y-%m-%d %H:%M"),
        "stage":            stage,
        "model":            model_name,
        "dataset_version":  dataset_version,
        "metrics":          metrics or {},
        "params":           params or {},
        "tags":             tags or [],
        "status":           "completed",
    }
    runs = _load_runs()
    runs.insert(0, entry)
    _save_runs(runs)
    return run_id


def get_runs_df() -> pd.DataFrame:
    runs = _load_runs()
    if not runs:
        return pd.DataFrame(columns=["run_id", "timestamp", "stage", "model", "dataset_version", "status"])

    rows = []
    for r in runs:
        row = {
            "run_id":          r["run_id"],
            "timestamp":       r["timestamp"],
            "stage":           r["stage"],
            "model":           r["model"],
            "dataset_version": r["dataset_version"],
            "status":          r["status"],
            "tags":            ", ".join(r.get("tags", [])),
        }
        for k, v in r.get("metrics", {}).items():
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def get_run_detail(run_id: str) -> dict:
    for r in _load_runs():
        if r["run_id"] == run_id:
            return r
    return {}


def clear_runs():
    _save_runs([])
