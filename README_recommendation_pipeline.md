# VoyageAnalytics: Travel Recommendation Pipeline

This README documents the recommendation-modeling part of **VoyageAnalytics**: how the raw travel records are transformed into journey-level datasets, how the destination recommenders are run and compared, and how to execute the workflow from one command.

The orchestration entry point is `run_recommendation_pipeline.py`. It runs the existing project scripts in sequence; it does **not** replace their modeling logic.

## 1. What this part of the project does

The recommendation workflow is designed to explore whether a traveller's previous journeys, travel dates, and destination patterns can help recommend a future destination.

At a high level, the pipeline:

1. Reads the flight and hotel records.
2. Constructs a journey-level dataset so the modeling code can work with one row per journey rather than treating every flight leg as an independent trip.
3. Creates chronological training and test datasets.
4. Runs baseline recommenders and destination-classification/ranking experiments.
5. Evaluates alternative chronological split strategies.
6. Analyses recurring user destination/month patterns.
7. Writes prediction tables, metric tables, trained-model artifacts where the individual scripts support them, and seasonality reports.

The pipeline prints each stage's console output. It stops immediately if a stage fails, rather than continuing and producing a confusing pile of partial results.

## 2. Repository layout

Place the files in the project root, with the raw CSV files and Python scripts alongside the pipeline runner:

```text
VoyageAnalytics/
├── run_recommendation_pipeline.py
├── prepare_travel_data.py
├── baseline_personalized_destination.py
├── baseline_destination.py
├── train_destination_model.py
├── train_destination_ranker.py
├── evaluate_destination_splits.py
├── analyze_user_seasonality.py
├── find_top_seasonal_travellers.py
├── users.csv
├── flights.csv
├── hotels.csv
└── travel_recommender_outputs/       # created/filled by the scripts
```

The app and EDA scripts may also live in the repository, but they are not launched by this orchestration script.

**Important:** filenames and paths are expected to match exactly. The pipeline uses paths relative to its own location and runs child scripts with the project root as the working directory. If your repository has a different layout, update the paths or move the files before running it.

## 3. Input data

The pipeline expects the following files in the project root.

| File | Role |
|---|---|
| `flights.csv` | Flight records, including journey/user identifiers, origin and destination, fare, travel time, distance, flight class, agency, and date. |
| `hotels.csv` | Hotel-booking records associated with journeys, including destination/place, days, price/total, user and journey identifiers, and date. |
| `users.csv` | User metadata such as the user code and profile fields. Required by the seasonal-traveller reporting stage. |

The journey preparation stage requires `flights.csv` and `hotels.csv`. The default full run also requires `users.csv` because `find_top_seasonal_travellers.py` uses it to enrich its report. If you intentionally skip seasonal reports, `users.csv` is not checked by the pipeline runner.

Do not rename columns casually. Individual scripts expect particular column names, or have their own column-resolution logic. Keep the raw files consistent with the versions used during development.

## 4. Pipeline stages

The default run executes these stages in order.

### Stage 1: Prepare journey datasets

**Script:** `prepare_travel_data.py`

Combines the flight and hotel data into a journey-level table and creates the core datasets used by downstream experiments:

- `travel_recommender_outputs/journeys.csv`
- `travel_recommender_outputs/journeys_train.csv`
- `travel_recommender_outputs/journeys_test.csv`

The train/test datasets are split chronologically so future journeys are evaluated after earlier journeys. This is more realistic for a recommendation task than randomly distributing records across train and test, which can allow information from the future to leak into training.

The pipeline checks that all three files exist after this stage. If one is missing, execution stops before training starts.

### Stage 2: Personalized destination baseline

**Script:** `baseline_personalized_destination.py`

Runs a personalized baseline using historical destination behaviour. It provides a useful reference point: more complex machine-learning models should be compared with this baseline, not merely celebrated for having an impressive acronym.

The script writes comparison, prediction, and metric CSV files under `travel_recommender_outputs/`. The exact metric values depend on the current input data and split, so they should be read from the generated reports rather than hard-coded into documentation.

### Stage 3: Destination baseline

**Script:** `baseline_destination.py`

Runs the additional/older destination baseline experiment. It is retained for comparison with the personalized baseline and later models.

This stage can be skipped with `--skip-legacy-models`.

### Stage 4: Destination classifier

**Script:** `train_destination_model.py`

Trains a Random Forest destination classifier using features built from journey and user-history information. It evaluates predictions on the prepared test data and writes its comparison/prediction outputs under `travel_recommender_outputs/`.

This is an overlapping comparison experiment rather than the only or automatically best model. Compare its results against the personalized baseline and candidate ranker.

This stage can be skipped with `--skip-legacy-models`.

### Stage 5: Candidate Random Forest ranker and seasonal experiment

**Script:** `train_destination_ranker.py`

Builds candidate-destination examples and trains a Random Forest model to score candidate destinations. It evaluates the ranking using top-k recommendation metrics and saves a serialized model with `joblib`, along with metric and prediction outputs.

The same script also evaluates a seasonal recommendation variant and writes a comparison between the approaches. Seasonal behavior should be treated as an experiment: repeated trips to the same destination in the same month can suggest a recurring pattern, but do not by themselves prove a seasonal preference.

This is the main candidate-ranking experiment in the current workflow. Its output should still be compared against the simpler personalized baseline on the same evaluation split.

### Stage 6: Chronological split evaluation

**Script:** `evaluate_destination_splits.py`

Compares recommendation approaches under different train/test splitting strategies, including:

- **Global chronological split:** trains on earlier journeys and tests on later journeys.
- **User-specific chronological split:** holds out later journeys for each user, where possible.
- **Popularity fallback comparison:** evaluates a global popularity ranking alongside personalized and Random Forest approaches.

The script writes split datasets and evaluation metrics to `travel_recommender_outputs/`. It is intended to help determine whether a model's apparent performance depends on how the data is split.

Use the same test set when comparing models. Otherwise, differences in scores can be caused by different test examples rather than a better recommender.

### Stage 7: User seasonality analysis

**Script:** `analyze_user_seasonality.py`

Summarizes each user's travel history and identifies recurring destination/month combinations using the training journeys. It writes CSV summaries under `travel_recommender_outputs/`, including user-level seasonality, recurring destination/month patterns, and monthly trip counts.

The current pattern definition flags a user/destination/month combination when it appears at least twice across at least two different years. This is a descriptive rule, not a statistical proof that the month caused the travel.

### Stage 8: Top seasonal-traveller report

**Script:** `find_top_seasonal_travellers.py`

Combines the prepared training journeys with `users.csv`, ranks or summarizes recurring seasonal travel patterns, and creates a report with CSV/text/plot outputs in the `seasonal_travellers/` directory by default.

This stage is useful for inspecting examples and communicating patterns. It should not be treated as a model evaluation metric.

## 5. Install the environment

Use Python 3.10 or newer if that matches the environment used for the rest of the project. The orchestration script itself uses only the Python standard library, but the child scripts require the libraries used by their modeling and reporting code.

From the project root, create and activate a virtual environment.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the dependencies required by the scripts. The modeling and reporting code uses packages including:

```bash
python -m pip install pandas numpy scikit-learn joblib matplotlib
```

If the project has a maintained `requirements.txt`, prefer installing from it instead:

```bash
python -m pip install -r requirements.txt
```

If a script imports a package not listed above, install that package in the same active environment. The pipeline launches each child script with the same Python interpreter used to run `run_recommendation_pipeline.py`, so activating the correct environment before execution matters.

## 6. Run the full pipeline

1. Put the raw CSVs and all required scripts in the project root.
2. Activate the project's virtual environment.
3. From the project root, run:

```bash
python run_recommendation_pipeline.py
```

The runner validates required files, prints the planned stages, executes each stage in sequence, checks the journey datasets after preparation, and stops on the first error.

A successful run ends with a completion message and points to `travel_recommender_outputs/`. Read each stage's console output and inspect its generated metrics and artifacts.

### Preview the commands without running the stages

```bash
python run_recommendation_pipeline.py --dry-run
```

The dry run validates the required raw inputs and selected script files, prints the planned commands, and does not execute the child scripts. It still expects the files needed for the selected configuration to exist.

### Skip seasonal reporting

```bash
python run_recommendation_pipeline.py --skip-seasonal-reports
```

This skips both `analyze_user_seasonality.py` and `find_top_seasonal_travellers.py`. The runner will not require `users.csv` for this run.

### Skip the older/overlapping baseline experiments

```bash
python run_recommendation_pipeline.py --skip-legacy-models
```

This skips `baseline_destination.py` and `train_destination_model.py`. The personalized baseline, candidate ranker, split evaluation, and (unless separately disabled) seasonal reports still run.

### Skip both optional groups

```bash
python run_recommendation_pipeline.py \
  --skip-seasonal-reports \
  --skip-legacy-models
```

### Run an individual stage

For debugging, each component can be run independently from the project root. For example:

```bash
python prepare_travel_data.py
python baseline_personalized_destination.py
python train_destination_ranker.py
python evaluate_destination_splits.py
python analyze_user_seasonality.py
```

Run preparation first whenever the raw input data changes, because downstream scripts read the prepared journey CSVs.

## 7. Outputs and where to look

Most model and evaluation outputs are written to `travel_recommender_outputs/`. Depending on the stage and current script version, outputs include:

| Output | Purpose |
|---|---|
| `journeys.csv` | Complete prepared journey-level dataset. |
| `journeys_train.csv` | Chronologically earlier journeys used for training/history. |
| `journeys_test.csv` | Held-out journeys used for evaluation. |
| `destination_model_comparison.csv` | Comparison output from the destination classifier experiment. |
| `random_forest_destination_predictions.csv` | Prediction output from the destination classifier. |
| `candidate_ranker_metrics.csv` | Metrics for the candidate Random Forest ranker. |
| `candidate_ranker_predictions.csv` | Candidate ranker prediction output. |
| `seasonal_destination_metrics.csv` | Evaluation metrics for the seasonal recommendation variant. |
| `seasonal_destination_predictions.csv` | Predictions from the seasonal variant. |
| `destination_ranker_comparison.csv` | Comparison of candidate-ranking and seasonal approaches. |
| Seasonality CSV reports | User-level summaries and recurring destination/month patterns. |
| Serialized ranker model | Saved model artifact from the candidate ranker, created by `joblib`. |

The exact file list may change as the scripts evolve. The most reliable inventory is the contents of `travel_recommender_outputs/` after a successful run. The runner does not guarantee that every experiment exports a serialized model; check the individual stage's code and console output before assuming an artifact exists.

The top seasonal-traveller report is written to `seasonal_travellers/` by default, including `seasonal_traveller_patterns.csv` and report/plot outputs.

## 8. How to interpret the evaluation

The recommendation scripts use ranking-oriented metrics, including:

- **Top-1 accuracy:** fraction of test journeys where the first recommendation matches the actual destination.
- **Hit Rate@5 / Hit@5:** fraction of test journeys where the actual destination appears anywhere in the top five recommendations.
- **MRR@5 (Mean Reciprocal Rank at 5):** rewards the model more when the correct destination appears near the top of the list. A correct destination at rank 1 contributes more than one at rank 5.
- **Coverage:** fraction of possible destinations that the recommender is able to recommend across the evaluated examples, where reported.

Use the same split and metric implementation when comparing models. A small metric improvement is not automatically meaningful; inspect sample counts, subgroup behavior, and whether the improvement holds for users with different amounts of history.

### Known interpretation limits

- A personalized baseline can outperform a more complex model. That is a valid result, not a reason to hide the baseline.
- The chronological test evaluates future journeys relative to a fixed training history. It does not necessarily simulate an online system that updates its history after every new booking.
- Recurring destination/month patterns are descriptive signals, not proof of causal seasonality.
- The hotel data currently has only one distinct hotel name per destination. Therefore, a destination-to-hotel result is effectively a lookup, not a meaningful personalized hotel-ranking evaluation. Do not claim that this data establishes a strong hotel recommendation model.
- Historical booking data may reflect limited coverage and user behavior in the dataset, not all possible travel preferences. Recommendations should be described accordingly.

## 9. Troubleshooting

### `Missing raw input dataset` or `Missing pipeline script`

The runner could not find a required file in the project root.

- Check spelling and capitalization.
- Confirm that the files are next to `run_recommendation_pipeline.py`.
- If your repository uses another directory layout, update the paths in the runner and relevant child scripts.

### `journeys.csv`, `journeys_train.csv`, or `journeys_test.csv` is missing

`prepare_travel_data.py` completed without creating one or more expected files, or writes to a different output directory.

- Run `python prepare_travel_data.py` directly to see its full error.
- Confirm that its configured output directory is `travel_recommender_outputs/`.
- Check for missing columns, malformed dates, or inconsistent journey identifiers in the source CSVs.

### A stage fails and the pipeline stops

This is intentional. The error printed immediately before `PIPELINE STOPPED` identifies the command that failed.

Run that child script directly to get a shorter debugging loop, fix its error, and then rerun the full pipeline. Completed earlier stages may have already written outputs; rerunning preparation and training will typically refresh them.

### `ModuleNotFoundError`

Activate the same environment used for installation, then install the missing package:

```bash
python -m pip install <package-name>
```

Check that `python` points to the expected interpreter:

```bash
which python
python --version
python -m pip --version
```

On Windows, use `where python` instead of `which python`.

### Results look unchanged after editing the raw data

Rerun the full pipeline. Downstream model scripts use the prepared journey datasets, not necessarily the raw CSVs directly.

### Model file was not created

Not every stage exports a serialized model. The candidate ranker saves a model with `joblib`; other experiments may only write predictions and metrics. Check the relevant script rather than assuming all training scripts have the same artifact behavior.

## 10. Reproducibility and good practice

- Keep a stable copy of the raw CSVs used for reported results.
- Record the code revision, Python version, package versions, split strategy, and metrics with each experiment.
- Do not use the test set to choose features or repeatedly tune seasonal rules. Use training/validation data for development and reserve the test set for final evaluation.
- Compare models on identical test journeys and identical ranking metrics.
- Inspect subgroup performance and sample sizes, not only aggregate metrics.
- Keep generated datasets and model artifacts out of version control if they are large or contain sensitive data; use `.gitignore` or an appropriate data-storage strategy.
- Treat the pipeline runner as orchestration. The implementation and assumptions of each stage remain defined by the individual scripts.

## 11. Quick-start checklist

- [ ] `flights.csv`, `hotels.csv`, and (for full run) `users.csv` are in the project root.
- [ ] All scripts listed in the repository layout are present.
- [ ] The virtual environment is active and dependencies are installed.
- [ ] `python run_recommendation_pipeline.py --dry-run` completes.
- [ ] `python run_recommendation_pipeline.py` completes without a failed stage.
- [ ] Metrics and prediction files in `travel_recommender_outputs/` have been reviewed.
- [ ] Reported model comparisons use the same held-out data and metrics.
