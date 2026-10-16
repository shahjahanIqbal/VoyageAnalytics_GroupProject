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

## What it does not do yet

This is a deliberately basic first MLOps layer. It does **not** automatically deploy models, register/promote a production model, monitor live traffic or drift, trigger scheduled retraining, or guarantee that different model owners' input/output contracts are compatible. Those are later steps in the project flowchart.

It also does not alter the model's training logic. Each model script must still validate its data, train/evaluate correctly, and write its metrics and artifacts.

**New in this version:** MLflow tracking commands for the new multi-model destination recommender, hotel recommender, and inference pipeline.

## 1. Install

Run these commands from the project root in the same virtual environment used for the models:

```bash
python -m pip install mlflow
```

The model-specific dependencies must also be installed. For example, the existing destination ranker uses pandas, NumPy, scikit-learn, and joblib; the group regression script also uses XGBoost.

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
  --metrics-file travel_recommender_outputs/destination_ranker_seasonal_comparison.csv \
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

The project flowchart assigns four workstreams: data engineering/EDA, flight price regression, hotel recommendation, and gender classification/platform integration. The wrapper is generic: each owner supplies their own command, metrics files, input files, and artifacts. No one needs to merge model internals just to log experiments.

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

The uploaded regression script reads `data/train.csv`, `data/val.csv`, and `data/test.csv`; it writes `outputs/regression_results.csv` and `outputs/regression_test_results.csv`, and saves a model under `models/`.

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

This example assumes the regression script's configured target and feature set are the intended flight-price task. The currently uploaded `regression.py` describes an attraction-rating regression target (`Rating`), so the owner must confirm the actual target before calling it *flight-price* regression.

### Member 3: Hotel recommendation

Use the hotel model owner's actual script and its actual output paths. For the uploaded history-based hotel recommender, the pattern is:

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

If the group's hotel recommender instead uses `recommendation.py` and the `data/`/`outputs/` structure, use that script's own metric and model paths instead. The command must match the model owner's implementation.

### Member 4: Gender classification and platform integration

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

Confirm the target column is actually gender before describing this as gender classification. The currently uploaded `classification.py` is configured around a visit-mode target (`VisitMode`), so the script name alone is not evidence that it predicts gender.

### Member 5: Multi-model destination recommender (NEW)

Tracks the new `train_destination_models.py` which trains 3 models (Random Forest, Gradient Boosting, Logistic Regression) on 2 split strategies (global chronological, user-specific chronological) with seasonal ML features.

```bash
python mlops_runner.py \
  --experiment-name VoyageAnalytics \
  --run-name destination-multi-model \
  --input-file travel_recommender_outputs/journeys_train.csv \
  --input-file travel_recommender_outputs/journeys_test.csv \
  --input-file travel_recommender_outputs/user_train.csv \
  --input-file travel_recommender_outputs/user_test.csv \
  --input-file travel_recommender_outputs/recurring_user_destination_month.csv \
  --input-file travel_recommender_outputs/user_monthly_trip_counts.csv \
  --metrics-file model_outputs/destination_model_comparison.csv \
  --artifact model_outputs/best_destination_model.joblib \
  --artifact model_outputs/destination_all_predictions.csv \
  --param models=random_forest,gradient_boosting,logistic_regression \
  --param splits=global_80_20,user_80_20 \
  --param seasonal_features=true \
  -- python train_destination_models.py
```

### Member 6: Hotel recommender ML model (NEW)

Tracks the new hotel recommender pipeline: data preparation + model training.

**Data preparation:**
```bash
python mlops_runner.py \
  --experiment-name VoyageAnalytics \
  --run-name hotel-data-prep \
  --input-file travel_recommender_outputs/journeys.csv \
  --input-file hotels.csv \
  --input-file users.csv \
  --artifact model_outputs/hotel_train.csv \
  --artifact model_outputs/hotel_test.csv \
  --artifact model_outputs/hotel_training_data.csv \
  --param stage=hotel_data_preparation \
  -- python prepare_hotel_data.py
```

**Model training:**
```bash
python mlops_runner.py \
  --experiment-name VoyageAnalytics \
  --run-name hotel-recommender-ml \
  --input-file model_outputs/hotel_train.csv \
  --input-file model_outputs/hotel_test.csv \
  --metrics-file model_outputs/hotel_model_comparison.csv \
  --artifact model_outputs/best_hotel_model.joblib \
  --artifact model_outputs/hotel_predictions_linear_regression.csv \
  --artifact model_outputs/hotel_predictions_random_forest.csv \
  --artifact model_outputs/hotel_predictions_gradient_boosting.csv \
  --param target=hotel_total \
  --param models=linear_regression,random_forest,gradient_boosting \
  -- python train_hotel_model.py
```

### Member 7: End-to-end inference pipeline (NEW)

Tracks the inference pipeline that combines destination and hotel recommenders.

```bash
python mlops_runner.py \
  --experiment-name VoyageAnalytics \
  --run-name inference-pipeline \
  --input-file model_outputs/best_destination_model.joblib \
  --input-file model_outputs/best_hotel_model.joblib \
  --input-file travel_recommender_outputs/journeys_with_hotels.csv \
  --input-file travel_recommender_outputs/user_train.csv \
  --artifact model_outputs/inference_result_*.json \
  --param pipeline=destination_to_hotel \
  -- python run_recommendation_inference.py --user_code 0 --origin "Recife (PE)" --departure_date "2024-07-15"
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

The wrapper logs the parameters you explicitly pass. It does not introspect every model hyperparameter from a child script automatically.

## 6. How metrics are logged

For every declared CSV, the runner reads numeric columns and logs them to MLflow. For a one-row metrics file, a column such as `hit_rate_at_5` is logged as a metric named like:

```text
candidate_ranker_metrics.hit_rate_at_5
```

For a multi-row CSV, each row gets a suffix such as `.row_1`, `.row_2`, and so on. Text model labels such as `Model` are recorded as tags when recognized. The CSV itself is uploaded as an artifact so the full table remains available.

For meaningful comparisons, all model owners should document the metric definitions and ensure higher/lower is better is understood. For example, higher Hit@5 is better, while lower MAE is better.

## 7. Team conventions to agree on

Before multiple members start logging runs, agree on:

1. One shared experiment name: `VoyageAnalytics`.
2. Consistent run names: `task-model-version` or `task-model-date`.
3. Standard metric names and definitions.
4. A small model contract: `train`, `evaluate`, `predict/recommend`, `save`, `load`.
5. Input data versions and feature/schema versions.
6. A consistent location for each model's metrics and serialized artifact.
7. Which model is approved for the app, and who approves a replacement.
8. Whether tracking is local for development or backed by a shared MLflow server.

Do not log personally identifying data or upload raw datasets as artifacts unless the team has explicitly decided that the storage and access controls are appropriate.

## 8. Suggested next steps after this starter

1. Make each model save a reusable model artifact and a feature/schema manifest.
2. Add automated tests for input schema, prediction shape, and metric calculations.
3. Add a CI job that runs tests and a small smoke-training run.
4. Add an inference API for each model owner or one API gateway that routes to each model service.
5. Add model approval/registry workflow only after the team agrees on validation gates.
6. Monitor service errors, input distributions, prediction distributions, and delayed ground-truth metrics.
7. Trigger retraining with a scheduler only when there is a defined data-refresh cadence or monitoring threshold.

The intended first milestone is **experiment tracking and reproducibility**, not a claim that the project already has full production MLOps.
