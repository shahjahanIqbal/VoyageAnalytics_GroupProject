#!/usr/bin/env python3
"""
Run the VoyageAnalytics travel-recommendation pipeline end to end.

Expected project layout:
    project/
    ├── run_recommendation_pipeline.py
    ├── prepare_travel_data.py
    ├── baseline_destination.py
    ├── baseline_personalized_destination.py
    ├── train_destination_model.py
    ├── train_destination_ranker.py
    ├── evaluate_destination_splits.py
    ├── analyze_user_seasonality.py
    ├── find_top_seasonal_travellers.py
    ├── users.csv
    ├── flights.csv
    └── hotels.csv

The script runs each existing project script as a separate process, using the same Python interpreter as this pipeline. It stops at the first failed stage.

Run from the project root:
    python run_recommendation_pipeline.py

"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "travel_recommender_outputs"

# Required raw files for prepare_travel_data.py, based on the project layout.
RAW_DATA_FILES = ("flights.csv", "hotels.csv")

# These scripts produce the prepared journey datasets.
PREPARATION_SCRIPT = "prepare_travel_data.py"

# Main model/evaluation stages. Keep names aligned with the repository.
MAIN_STAGES = [
    ("Personalized baseline", "baseline_personalized_destination.py"),
    ("Destination baseline", "baseline_destination.py"),
    ("Destination classifier", "train_destination_model.py"),
    ("Candidate Random Forest ranker", "train_destination_ranker.py"),
    ("Chronological split evaluation", "evaluate_destination_splits.py"),
]

SEASONAL_STAGES = [
    ("User seasonality analysis", "analyze_user_seasonality.py"),
    ("Top seasonal traveller report", "find_top_seasonal_travellers.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare journey datasets, train recommenders, and run evaluations."
    )
    parser.add_argument(
        "--skip-seasonal-reports",
        action="store_true",
        help="Skip the user seasonality and top-traveller reporting scripts.",
    )
    parser.add_argument(
        "--skip-legacy-models",
        action="store_true",
        help=(
            "Skip baseline_destination.py and train_destination_model.py, "
            "which are older/overlapping comparison experiments."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands and validate required scripts/data without running them.",
    )
    return parser.parse_args()


def require_file(path: Path, purpose: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {purpose}: {path}\n"
            "Check the filename and place the file in the project root, "
            "or adjust the paths in run_recommendation_pipeline.py."
        )


def run_stage(label: str, script_name: str, dry_run: bool = False) -> None:
    script_path = ROOT / script_name
    require_file(script_path, "pipeline script")

    command = [sys.executable, str(script_path)]
    print(f"\n{'=' * 72}\nSTAGE: {label}\nCOMMAND: {' '.join(command)}\n{'=' * 72}", flush=True)

    if dry_run:
        return

    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, check=False)
    elapsed = time.perf_counter() - started

    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)

    print(f"Completed: {label} ({elapsed:.1f}s)", flush=True)


def validate_prepared_data() -> None:
    expected = [
        OUTPUT_DIR / "journeys.csv",
        OUTPUT_DIR / "journeys_train.csv",
        OUTPUT_DIR / "journeys_test.csv",
    ]
    missing = [p for p in expected if not p.is_file()]
    if missing:
        paths = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "The preparation stage finished, but expected journey datasets are missing:\n"
            f"{paths}\n"
            "Check the output directory configured in prepare_travel_data.py."
        )


def main() -> int:
    args = parse_args()
    print(f"Project root: {ROOT}")
    print(f"Output directory expected: {OUTPUT_DIR}")

    # Check all raw inputs before starting, to fail early rather than halfway through.
    for filename in RAW_DATA_FILES:
        require_file(ROOT / filename, "raw input dataset")

    # Check the preparation script and all selected downstream scripts first.
    require_file(ROOT / PREPARATION_SCRIPT, "journey preparation script")

    stages = []
    stages.append(("Prepare journey datasets", PREPARATION_SCRIPT))

    main_stages = list(MAIN_STAGES)
    if args.skip_legacy_models:
        main_stages = [
            item for item in main_stages
            if item[1] not in {"baseline_destination.py", "train_destination_model.py"}
        ]
    stages.extend(main_stages)

    if not args.skip_seasonal_reports:
        # The seasonal traveller report uses users.csv as well as journeys_train.csv.
        require_file(ROOT / "users.csv", "user lookup dataset")
        stages.extend(SEASONAL_STAGES)

    # Validate script files before any stage is launched.
    for _, script_name in stages:
        require_file(ROOT / script_name, "pipeline script")

    print("\nPlanned stages:")
    for index, (label, script_name) in enumerate(stages, start=1):
        print(f"  {index}. {label}: {script_name}")

    if args.dry_run:
        print("\nDry run only. No scripts were executed.")
        for label, script_name in stages:
            run_stage(label, script_name, dry_run=True)
        return 0

    pipeline_started = time.perf_counter()

    for index, (label, script_name) in enumerate(stages, start=1):
        run_stage(label, script_name)

        if script_name == PREPARATION_SCRIPT:
            validate_prepared_data()
            print("Verified journeys.csv, journeys_train.csv, and journeys_test.csv.")

    elapsed = time.perf_counter() - pipeline_started
    print(f"\nPipeline completed successfully in {elapsed / 60:.1f} minutes.")
    print(f"Journey datasets: {OUTPUT_DIR}")
    print(
        "Review each stage's console output for its specific model metrics and "
        "generated artifacts. The script does not assume every model script "
        "exports a serialized model."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(f"\nPIPELINE ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except subprocess.CalledProcessError as exc:
        print(
            f"\nPIPELINE STOPPED: command exited with status {exc.returncode}:\n"
            f"  {' '.join(map(str, exc.cmd))}\n"
            "Fix that stage's error before rerunning. Later stages were not executed.",
            file=sys.stderr,
        )
        raise SystemExit(exc.returncode or 1)
