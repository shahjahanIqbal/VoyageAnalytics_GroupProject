# VoyageAnalytics MLOps Starter

This is a small, reusable MLflow tracking layer for the VoyageAnalytics group project. It wraps each member's existing training/evaluation script without requiring everyone to rewrite their model.

## What it does

For each experiment, `mlops_runner.py`:

1. Starts an MLflow run under a shared experiment name.
2. Records the command, Python version, platform, optional parameters and tags.
3. Calculates SHA-256 fingerprints for declared input files, so runs can be tied to the exact input-data versions.
4. Executes the member's existing training/evaluation script.
5. Captures the child process console output as a log artifact.
6. Reads the declared metrics CSV files and logs numeric columns as MLflow metrics.
7. Uploads selected output files, reports, and serialized models as MLflow artifacts.
8. Marks the run as completed or failed and records runtime/exit code.

The default tracking store is a local `mlruns/` directory. MLflow's UI lets the team compare experiments and inspect each run.

## 1. Install

Run these commands from the project root in the same virtual environment used for the models:

```bash
python -m pip install mlflow
```

The model-specific dependencies must also be installed. 

Check the runner:

```bash
python mlops_runner.py --help
```

## 2. Run the current travel destination ranker with MLflow

First prepare the journey datasets if they do not already exist:

```bash
python prepare_travel_data.py
```

Then run the candidate destination ranker under MLflow:

```bash
python mlops_runner.py \
  --experiment-name VoyageAnalytics \
  --run-name destination-ranker \
  --input-file travel_recommender_outputs/journeys_train.csv \
  --input-file travel_recommender_outputs/journeys_test.csv \
  --metrics-file travel_recommender_outputs/candidate_ranker_metrics.csv \
  --metrics-file travel_recommender_outputs/seasonal_destination_metrics.csv \
  --metrics-file travel_recommender_outputs/destination_ranker_comparison.csv \
  --artifact travel_recommender_outputs/candidate_ranker.joblib \
  --artifact travel_recommender_outputs/candidate_ranker_predictions.csv \
  --artifact travel_recommender_outputs/seasonal_destination_predictions.csv \
  -- python train_destination_ranker.py
```

Paths are relative to the project root. Only specify metrics/artifacts that the selected script actually creates. If your local version writes a different filename, change the relevant option.

The wrapper runs the training script first, then reads the requested CSVs. If a metrics file does not exist, the run is marked `completed_metrics_missing` (or failed if the child process failed) and a warning is printed.

## 3. Open the MLflow UI

In a second terminal, from the same project root and environment:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open the local URL printed by MLflow, usually `http://127.0.0.1:5000`.

Use the `VoyageAnalytics` experiment to compare run names, metrics, parameters, status, and artifacts. Keep the MLflow UI running in that terminal while viewing it.

If the team uses a remote MLflow tracking server, set `MLFLOW_TRACKING_URI` or pass `--tracking-uri`. Do not point different teammates at unrelated local tracking stores if you want a shared experiment history.

## 4. How each group member can use the same wrapper

The wrapper is generic: each owner supplies their own command, metrics files, input files, and artifacts. No one needs to merge model internals just to log experiments.

### Member 1: Data engineering and EDA

The data owner can track the data-preparation/validation script as an experiment, recording input fingerprints and uploading validation reports or processed-data manifests. If the script does not emit numeric metric CSVs, omit `--metrics-file` and track the generated report with `--artifact`.

Example pattern:

```bash
python mlops_runner.py \
  --experiment-name VoyageAnalytics \
  --run-name data-validation \
  --input-file flights.csv \
  --input-file hotels.csv \
  --artifact travel_recommender_outputs/journeys.csv \
  --param stage=data_validation \
  -- python prepare_travel_data.py
```

Only use those input paths if that script actually consumes them. For a different data-preparation script, replace the command and paths.

### Member 2: Flight-price regression

```bash
python mlops_runner.py \
  --experiment-name VoyageAnalytics \
  --run-name flight-regression \
  --input-file data/train.csv \
  --input-file data/val.csv \
  --input-file data/test.csv \
  --metrics-file outputs/regression_results.csv \
  --metrics-file outputs/regression_test_results.csv \
  --artifact models/regression_model.joblib \
  --param task=regression \
  -- python regression.py
```

This example assumes the regression script's configured target and feature set are the intended flight-price task. 

### Member 3: Hotel recommendation

Use the actual script and its output paths. 

```bash
python mlops_runner.py \
  --experiment-name VoyageAnalytics \
  --run-name hotel-recommender \
  --input-file hotels.csv \
  --metrics-file travel_recommender_outputs/hotel_metrics_personalized.csv \
  --artifact travel_recommender_outputs/hotel_predictions_personalized.csv \
  --param task=hotel_recommendation \
  -- python train_hotel_recommender.py
```

### Member 4: Gender classification 

For the currently uploaded classification script, the pattern is:

```bash
python mlops_runner.py \
  --experiment-name VoyageAnalytics \
  --run-name classification \
  --input-file data/train.csv \
  --input-file data/val.csv \
  --input-file data/test.csv \
  --metrics-file outputs/classification_results.csv \
  --metrics-file outputs/classification_test_results.csv \
  --metrics-file outputs/classification_confusion_matrix.csv \
  --artifact models/classification_model.joblib \
  --param task=classification \
  -- python classification.py
```

## 5. CLI options

- `--experiment-name`: shared MLflow experiment name (default `VoyageAnalytics`).
- `--run-name`: readable name for this run.
- `--tracking-uri`: local or remote MLflow tracking URI.
- `--metrics-file PATH`: CSV whose numeric columns should be logged; repeat to add multiple files.
- `--artifact PATH`: file/directory to upload; repeat to add multiple artifacts.
- `--input-file PATH`: input to fingerprint with SHA-256 and file size; repeat as needed.
- `--param KEY=VALUE`: parameter metadata; repeat as needed.
- `--tag KEY=VALUE`: run tags; repeat as needed.
- `--project-root PATH`: working directory for the child script.
- `-- COMMAND`: command to run, after the options.

Example with parameters/tags:

```bash
python mlops_runner.py \
  --run-name ranker-experiment-02 \
  --param n_estimators=100 \
  --param min_samples_leaf=5 \
  --tag owner=member_4 \
  --tag model_family=random_forest \
  --metrics-file travel_recommender_outputs/candidate_ranker_metrics.csv \
  -- python train_destination_ranker.py
```

## 6. Metrics Logging

For every declared CSV, the runner reads numeric columns and logs them to MLflow. For a one-row metrics file, a column such as `hit_rate_at_5` is logged as a metric named like:

```text
candidate_ranker_metrics.hit_rate_at_5
```

For a multi-row CSV, each row gets a suffix such as `.row_1`, `.row_2`, and so on. Text model labels such as `Model` are recorded as tags when recognized. The CSV itself is uploaded as an artifact so the full table remains available.

For meaningful comparisons, all model owners should document the metric definitions and ensure higher/lower is better is understood. For example, higher Hit@5 is better, while lower MAE is better.
