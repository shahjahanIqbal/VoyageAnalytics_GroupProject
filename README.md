# VoyageAnalytics — Travel Recommendation & Flight Price Prediction

Group project for the DS/AI ML internship. The app combines:

- **Destination Planner** — ML destination recommendations (6 models: Random Forest / Gradient Boosting / Logistic Regression × 2 chronological split strategies) served by a Flask API, with seasonal personalization features.
- **Hotel Expenditure Prediction** — 3 regression models (Linear Regression / Random Forest / Gradient Boosting) predicting hotel spend per user-destination pair, with historical hotel lookup.
- **Flight Price Predictor** — Random Forest regressor predicting flight prices.

> The destination planner uses the Flask API in `api.py` (port 5001). The Streamlit interface (`app.py`, port 8501) calls that API and gracefully falls back to local heuristics when the API or models are missing.

## Important: no binary model files in this repository

The `.joblib` model files (including the flight model and encoders) are **generated locally** and are not committed. Follow the setup order below to produce every data file and model this project needs from the raw datasets.

You need these raw datasets in the project root (they are the "given" datasets):

| File | In repo? | Used by |
|---|---|---|
| `flights.csv` | yes | journey preparation, flight model |
| `hotels.csv` | no (supplied) | journey preparation, hotel models |
| `users.csv` | no (supplied) | seasonality analysis, hotel models |

## Setup order (do these in order)

### 1. Create the environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install mlflow                 # optional, only for experiment tracking
```

Place `flights.csv`, `hotels.csv`, and `users.csv` in the project root.

### 2. Generate the journey datasets and seasonal features

Run the full data-preparation and baseline pipeline:

```bash
python run_recommendation_pipeline.py
```

This runs, in order: `prepare_travel_data.py` → personalized/global baselines → `train_destination_model.py` → `train_destination_ranker.py` → chronological split evaluation → `analyze_user_seasonality.py` → `find_top_seasonal_travellers.py`.

Or run the required pieces individually (minimum needed for the app):

```bash
python prepare_travel_data.py            # -> journeys.csv, journeys_train.csv, journeys_test.csv
python evaluate_destination_splits.py    # -> global_train.csv, global_test.csv, user_train.csv, user_test.csv
python analyze_user_seasonality.py       # -> recurring_user_destination_month.csv, user_monthly_trip_counts.csv, user_seasonality_summary.csv
```

All outputs land in `travel_recommender_outputs/`.

### 3. Prepare the hotel training data

```bash
python prepare_hotel_data.py
```

Outputs (in `model_outputs/`): `hotel_train.csv`, `hotel_test.csv`, `hotel_training_data.csv`, `hotel_feature_columns.csv`.

### 4. Train the destination and hotel models

Stage-wise (recommended; also writes comparison and prediction CSVs):

```bash
python train_destination_models.py   # 6 destination models + seasonal features + candidates
python train_hotel_model.py          # 3 hotel models + metrics
```

Or the one-shot alternative (trains all 9 models in one command):

```bash
python train_all_models.py
```

Outputs (in `model_outputs/`): `dest_model_*.joblib`, `hotel_model_*.joblib`, `best_destination_model.joblib`, `best_hotel_model.joblib`, `dest_candidates.joblib`, `seasonal_features_*.joblib`, plus comparison/prediction CSVs.

### 5. Train the flight price model (for the Flight Price Predictor tab)

Open and run the notebook end to end:

```
flight price prediction.ipynb
```

It writes `flight_price_model.joblib`, `from_encoder.joblib`, `to_encoder.joblib`, `flightType_encoder.joblib`, and `agency_encoder.joblib` to the project root. If these files are missing, the Flight Price Predictor tab shows a clear error; the Destination Planner is unaffected.

### 6. Verify the pipeline

```bash
python evaluate_all_models.py
python run_recommendation_inference.py --user_code 0 --origin "Recife (PE)" --departure_date "2024-07-15"
```

`evaluate_all_models.py` prints a unified metrics report. `run_recommendation_inference.py` runs the full destination → hotel inference path and writes `model_outputs/inference_result_<user>_<date>.json`.

### 7. (Optional) Log the models with MLflow

```bash
pip install mlflow
python mlops_runner.py --help        # see README_MLOps_Starter.md for the exact commands
mlflow ui --backend-store-uri ./mlruns
```

`README_MLOps_Starter.md` contains ready-to-copy commands for the destination multi-model, hotel data prep, hotel model, and inference pipeline runs.

### 8. Run the application

Two processes:

```bash
python api.py            # Flask API  -> http://localhost:5001
streamlit run app.py     # Streamlit  -> http://localhost:8501
```

Or with Docker (build after step 5 so the trained `.joblib` files are present in the build context):

```bash
docker compose up --build
```

The container runs the API (5001) and Streamlit (8501) together; open http://localhost:8501.

## Repository layout (key files)

```
api.py                            Flask API serving all destination/hotel models
app.py                            Streamlit interface (calls the API, local fallbacks)
train_all_models.py               one-shot training for all destination + hotel models
train_destination_models.py       destination models (3 algorithms × 2 splits) + seasonal features
prepare_hotel_data.py             hotel training data preparation
train_hotel_model.py              hotel expenditure models
run_recommendation_inference.py   CLI end-to-end inference (destination -> hotel)
evaluate_all_models.py            unified evaluation report
run_recommendation_pipeline.py    orchestrates the legacy data/baseline pipeline
mlops_runner.py                   generic MLflow tracking wrapper
Dockerfile / docker-start.sh      container entrypoint (API + Streamlit)
README_recommendation_pipeline.md detailed pipeline documentation (stages 1-12)
README_MLOps_Starter.md           MLflow tracking instructions
```

## Contributing

Please branch the repo when contributing and include a detailed description of the changes made to the project. Once we test everything, the branches will be integrated to the main branch. Cheers!
