#!/usr/bin/env python3
"""
Small, reusable MLflow wrapper for VoyageAnalytics model scripts.

Run any member's existing training/evaluation script as a child process,
track the run in MLflow, log numeric metrics from CSVs, and save selected
artifacts and input-data fingerprints.

This is an experiment-tracking layer. It does not rewrite model code, choose
the best model automatically, or deploy/register a production model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import tempfile
from pathlib import Path
from typing import Iterable

import mlflow
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a VoyageAnalytics model script and track it with MLflow."
    )
    parser.add_argument(
        "--experiment-name",
        default="VoyageAnalytics",
        help="MLflow experiment name shared by the project.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Human-readable run name. Defaults to the command's script name.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help=(
            "MLflow tracking URI. Defaults to a local ./mlruns directory. "
            "Can also be set with MLFLOW_TRACKING_URI."
        ),
    )
    parser.add_argument(
        "--metrics-file",
        action="append",
        default=[],
        help="CSV containing numeric metrics to log; repeat this option as needed.",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help=(
            "File or directory to upload as an MLflow artifact; repeat as needed. "
            "Prefer model files, reports, and selected outputs over raw datasets."
        ),
    )
    parser.add_argument(
        "--input-file",
        action="append",
        default=[],
        help=(
            "Input file whose SHA-256 fingerprint and size should be logged; "
            "the file contents are not uploaded automatically."
        ),
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional experiment parameter; repeat as needed.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional MLflow tag; repeat as needed.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Working directory for the child command (default: current directory).",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run. Put it after --, e.g. -- python train_destination_ranker.py",
    )
    return parser.parse_args()


def parse_key_value(items: Iterable[str], option_name: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{option_name} expects KEY=VALUE, received: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{option_name} key cannot be empty.")
        parsed[key] = value
    return parsed


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_signature(path: Path) -> tuple[int, int] | None:
    """Return a lightweight signature used to detect stale output files."""
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def safe_mlflow_name(value: str) -> str:
    """Keep metric names readable and reasonably compatible with MLflow."""
    cleaned = "".join(ch if ch.isalnum() or ch in "._-/" else "_" for ch in value)
    return cleaned[:250] or "metric"


def log_metrics_csv(path: Path) -> int:
    """Log numeric cells from a metrics CSV. Returns the number logged."""
    frame = pd.read_csv(path)
    if frame.empty:
        print(f"WARNING: metrics file is empty: {path}")
        return 0

    count = 0
    for row_index, row in frame.iterrows():
        for column in frame.columns:
            value = row[column]
            if pd.isna(value):
                continue

            # Text labels (e.g. model names) are better represented as tags/params.
            if isinstance(value, str):
                if column.lower() in {"model", "model_name", "experiment", "variant"}:
                    tag_value = value[:500]
                    mlflow.set_tag(f"{path.stem}.{column}", tag_value)
                continue

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue

            if not pd.notna(numeric_value):
                continue

            metric_name = safe_mlflow_name(f"{path.stem}.{column}")
            if len(frame) > 1:
                metric_name = safe_mlflow_name(f"{metric_name}.row_{row_index + 1}")

            mlflow.log_metric(metric_name, numeric_value)
            count += 1

    mlflow.log_artifact(str(path), artifact_path="metrics")
    print(f"Logged {count} numeric metrics from {path}")
    return count


def log_selected_artifact(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Artifact does not exist: {path}")

    if path.is_file():
        mlflow.log_artifact(str(path), artifact_path="artifacts")
        return

    # Walk the selected directory and upload each file while preserving relative paths.
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        relative_parent = file_path.parent.relative_to(path)
        artifact_subdir = Path("artifacts") / path.name / relative_parent
        mlflow.log_artifact(str(file_path), artifact_path=artifact_subdir.as_posix())


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project root does not exist: {root}")

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError(
            "No child command supplied. Example: "
            "-- python train_destination_ranker.py"
        )

    # Resolve relative script paths relative to the project root, not this wrapper's location.
    command_display = shlex.join(command)
    default_script = command[1] if len(command) > 1 and Path(command[0]).name in {"python", "python3"} else command[0]
    run_name = args.run_name or Path(default_script).name.replace(".py", "")
    tracking_uri = args.tracking_uri or (root / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    params = parse_key_value(args.param, "--param")
    tags = parse_key_value(args.tag, "--tag")

    metric_paths = [resolve_path(root, p) for p in args.metrics_file]
    artifact_paths = [resolve_path(root, p) for p in args.artifact]
    input_paths = [resolve_path(root, p) for p in args.input_file]

    missing_inputs = [p for p in input_paths if not p.is_file()]
    if missing_inputs:
        raise FileNotFoundError(
            "Input file(s) not found:\n" + "\n".join(f"  - {p}" for p in missing_inputs)
        )

    # Log reproducibility metadata before running the training command.
    params.setdefault("child_command", command_display[:500])
    params.setdefault("python_version", platform.python_version())
    params.setdefault("platform", platform.platform())
    params.setdefault("project_root_name", root.name)

    started = time.perf_counter()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("project", "VoyageAnalytics")
        mlflow.set_tag("tracking_purpose", "training_evaluation")
        mlflow.set_tag("child_command", command_display[:500])
        mlflow.set_tag("run_status", "running")
        for key, value in tags.items():
            mlflow.set_tag(key, value)
        for key, value in params.items():
            mlflow.log_param(key, str(value)[:500])

        # Fingerprint inputs so runs can be compared against exact data versions.
        for index, path in enumerate(input_paths, start=1):
            mlflow.log_param(f"input_{index}_name", path.name[:250])
            mlflow.log_param(f"input_{index}_size_bytes", path.stat().st_size)
            mlflow.set_tag(f"input_{index}_sha256", sha256_file(path))
        if input_paths:
            mlflow.log_dict(
                {
                    str(path.relative_to(root) if path.is_relative_to(root) else path): {
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for path in input_paths
                },
                "input_manifest.json",
            )

        print(f"MLflow experiment: {args.experiment_name}")
        print(f"MLflow run ID: {run.info.run_id}")
        print(f"Tracking URI: {tracking_uri}")
        print(f"Working directory: {root}")
        print(f"Executing: {command_display}", flush=True)

        # Snapshot requested metric files so a stale CSV from an earlier run
        # cannot accidentally be logged as the result of this run.
        metric_signatures_before = {path: file_signature(path) for path in metric_paths}

        result = subprocess.run(
            command,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        elapsed = time.perf_counter() - started

        log_text = result.stdout or ""
        print(log_text, end="" if log_text.endswith("\n") else "\n")
        with tempfile.TemporaryDirectory(prefix="voyage_mlops_") as temp_dir:
            log_path = Path(temp_dir) / "child_process.log"
            log_path.write_text(log_text, encoding="utf-8", errors="replace")
            mlflow.log_artifact(str(log_path), artifact_path="logs")

        mlflow.log_metric("child_process_runtime_seconds", elapsed)
        mlflow.log_param("child_return_code", result.returncode)

        # Only log metric CSVs created or changed by this run.
        fresh_metrics = []
        missing_metrics = []
        for path in metric_paths:
            before = metric_signatures_before.get(path)
            after = file_signature(path)
            if after is not None and after != before:
                fresh_metrics.append(path)
            else:
                missing_metrics.append(path)

        for path in fresh_metrics:
            log_metrics_csv(path)
        for path in artifact_paths:
            if path.exists():
                log_selected_artifact(path)
            else:
                print(f"WARNING: requested artifact not found; skipped: {path}")

        if missing_metrics:
            mlflow.set_tag("run_status", "failed_missing_metrics")
            mlflow.set_tag(
                "missing_metrics_files",
                "; ".join(str(p) for p in missing_metrics)[:500],
            )

        if result.returncode != 0:
            mlflow.set_tag("run_status", "failed")
            mlflow.set_tag("failure_reason", "child_process_nonzero_exit")
            print(f"Training command failed with exit code {result.returncode}.")
            return result.returncode

        if missing_metrics:
            # The script may have succeeded but did not produce the requested evaluation file.
            mlflow.set_tag("run_status", "completed_metrics_missing")
            print(
                "WARNING: command completed, but one or more requested metric CSVs "
                "were not found. Check the paths and the child script's outputs."
            )
        else:
            mlflow.set_tag("run_status", "completed")

        print(f"Run finished in {elapsed:.1f}s.")
        print(f"Run ID: {run.info.run_id}")
        print(f"View results with: mlflow ui --backend-store-uri {tracking_uri}")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
        print(f"MLOPS RUNNER ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
