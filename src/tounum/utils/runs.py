"""Run directories and safe artifact writing.

Each training command writes into its own timestamped folder
(``artifacts/<task>/<timestamp>``), so that a new run never silently
overwrites the models or metrics of a previous one. This fixes the
notebooks' behaviour, where the captioning weights of the 20 000-photo
model were overwritten by the final model under the same file name.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def create_run_dir(artifacts_dir: Path, task: str, name: str | None = None) -> Path:
    """Create ``artifacts/<task>/<name or timestamp>``; refuse to reuse a folder."""
    run_name = name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = artifacts_dir / task / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory already exists and is not empty: {run_dir}. "
            "Choose another --run-name or delete it explicitly."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def latest_run(artifacts_dir: Path, task: str) -> Path:
    """Most recent run folder of a task (folders are named by timestamp)."""
    task_dir = artifacts_dir / task
    runs = sorted(p for p in task_dir.glob("*") if p.is_dir()) if task_dir.exists() else []
    if not runs:
        raise FileNotFoundError(f"No '{task}' run found in {task_dir}. Train the model first.")
    return runs[-1]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically (no half-written file if the process stops)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_to_jsonable(data), indent=2, ensure_ascii=False)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as tmp:
        tmp.write(payload)
    os.replace(tmp.name, path)


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, frame: pd.DataFrame, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)
