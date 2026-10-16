#!/usr/bin/env python3
"""
Pre-train and save all destination and hotel models for API use.
Run this once, then the API will load the saved models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

MODEL_DIR = Path("model_outputs")
DATA_DIR = Path("travel_recommender_outputs")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 5
RANDOM_STATE = 42

# Feature definitions
DEST_CATEGORICAL = ["userCode", "origin", "candidate"]
DEST_NUMERIC = [
    "month_sin", "month_cos", "day_of_week",
    "user_candidate_count", "user_route_count", "origin_candidate_count",
    "candidate_popularity", "user_history_count", "user_origin_history_count",
    "origin_history_count",
    "seasonal_trips", "seasonal_years", "user_month_total", "user_dest_total",
    "global_month_dest_rate", "seasonal_strength", "is_recurring_seasonal",
    "user_dest_month_share", "dest_month_popularity",
]
DEST_FEATURES = DEST_CATEGORICAL + DEST_NUMERIC

HOTEL_CATEGORICAL = ["user_preferred_flight_class", "user_preferred_agency", "user_dest_most_common_hotel"]
HOTEL_NUMERIC = [
    "user_past_trip_count", "user_past_total_expenditure", "user_past_avg_expenditure",
    "user_past_hotel_expenditure", "user_past_avg_hotel_expenditure",
    "user_past_flight_expenditure", "user_past_avg_flight_expenditure",
    "user_unique_destinations", "user_unique_origins",
    "user_dest_past_hotel_stays", "user_dest_avg_hotel_price", "user_dest_avg_hotel_total",
    "dest_avg_hotel_price", "dest_avg_hotel_total", "dest_avg_hotel_days", "dest_hotel_count",
    "departure_month", "departure_month_sin", "departure_month_cos", "departure_day_of_week",
]
HOTEL_FEATURES = HOTEL_CATEGORICAL + HOTEL_NUMERIC

# ============================================================
# SEASONAL FEATURES
# ============================================================
def build_seasonal_features(train_df):
    train_df = train_df.copy()
    train_df["departure_date"] = pd.to_datetime(train_df["departure_date"])
    train_df["month"] = train_df["departure_date"].dt.month
    train_df["year"] = train_df["departure_date"].dt.year
    
    seasonal_counts = train_df.groupby(["userCode", "destination", "month"]).agg(
        seasonal_trips=("travelCode", "count"),
        seasonal_years=("year", "nunique")
    ).reset_index()
    
    user_month_totals = train_df.groupby(["userCode", "month"]).agg(
        user_month_total=("travelCode", "count")
    ).reset_index()
    
    user_dest_totals = train_df.groupby(["userCode", "destination"]).agg(
        user_dest_total=("travelCode", "count")
    ).reset_index()
    
    global_month_dest = train_df.groupby(["month", "destination"]).agg(
        global_month_dest_count=("travelCode", "count")
    ).reset_index()
    global_month_total = train_df.groupby("month").agg(
        global_month_total=("travelCode", "count")
    ).reset_index()
    global_month_dest = global_month_dest.merge(global_month_total, on="month")
    global_month_dest["global_month_dest_rate"] = (
        global_month_dest["global_month_dest_count"] / global_month_dest["global_month_total"]
    )
    global_month_dest = global_month_dest.rename(columns={"destination": "candidate"})
    
    return {
        "seasonal_counts": seasonal_counts,
        "user_month_totals": user_month_totals,
        "user_dest_totals": user_dest_totals,
        "global_month_dest": global_month_dest,
    }

def add_seasonal_features(df, feature_dict, smoothing=5.0):
    df = df.copy()
    seasonal = feature_dict["seasonal_counts"].rename(columns={
        "userCode": "userCode", "destination": "candidate", "month": "month"
    })
    df = df.merge(
        seasonal[["userCode", "candidate", "month", "seasonal_trips", "seasonal_years"]],
        on=["userCode", "candidate", "month"], how="left"
    )
    df["seasonal_trips"] = df["seasonal_trips"].fillna(0).astype(int)
    df["seasonal_years"] = df["seasonal_years"].fillna(0).astype(int)
    
    user_month = feature_dict["user_month_totals"].rename(columns={"userCode": "userCode", "month": "month"})
    df = df.merge(user_month, on=["userCode", "month"], how="left")
    df["user_month_total"] = df["user_month_total"].fillna(0).astype(int)
    
    user_dest = feature_dict["user_dest_totals"].rename(columns={"userCode": "userCode", "destination": "candidate"})
    df = df.merge(user_dest, on=["userCode", "candidate"], how="left")
    df["user_dest_total"] = df["user_dest_total"].fillna(0).astype(int)
    
    global_md = feature_dict["global_month_dest"][["month", "candidate", "global_month_dest_rate"]]
    df = df.merge(global_md, on=["month", "candidate"], how="left")
    df["global_month_dest_rate"] = df["global_month_dest_rate"].fillna(0)
    
    df["seasonal_strength"] = (df["seasonal_trips"] + smoothing) / (df["user_month_total"] + smoothing)
    df["is_recurring_seasonal"] = ((df["seasonal_trips"] >= 2) & (df["seasonal_years"] >= 2)).astype(int)
    df["user_dest_month_share"] = df["seasonal_trips"] / (df["user_dest_total"] + 1)
    df["dest_month_popularity"] = df["global_month_dest_rate"]
    return df

def make_training_candidates(journeys, candidate_destinations, seasonal_features):
    journeys = journeys.sort_values(["departure_date", "travelCode"]).copy()
    
    user_destination = {}
    user_route = {}
    origin_destination = {}
    destination_counts = {}
    user_trip_counts = {}
    user_origin_counts = {}
    origin_counts = {}
    
    records = []
    
    for date, daily in journeys.groupby("departure_date", sort=True):
        for row in daily.itertuples(index=False):
            user = row.userCode
            origin = row.origin
            actual = row.destination
            month = date.month
            weekday = date.dayofweek
            
            for candidate in candidate_destinations:
                records.append({
                    "travelCode": row.travelCode,
                    "userCode": user,
                    "origin": origin,
                    "candidate": candidate,
                    "month_sin": np.sin(2 * np.pi * month / 12),
                    "month_cos": np.cos(2 * np.pi * month / 12),
                    "day_of_week": weekday,
                    "month": month,
                    "user_candidate_count": user_destination.get((user, candidate), 0),
                    "user_route_count": user_route.get((user, origin, candidate), 0),
                    "origin_candidate_count": origin_destination.get((origin, candidate), 0),
                    "candidate_popularity": destination_counts.get(candidate, 0),
                    "user_history_count": user_trip_counts.get(user, 0),
                    "user_origin_history_count": user_origin_counts.get((user, origin), 0),
                    "origin_history_count": origin_counts.get(origin, 0),
                    "label": int(candidate == actual),
                })
        
        for row in daily.itertuples(index=False):
            user = row.userCode
            origin = row.origin
            destination = row.destination
            
            user_destination[(user, destination)] = user_destination.get((user, destination), 0) + 1
            user_route[(user, origin, destination)] = user_route.get((user, origin, destination), 0) + 1
            origin_destination[(origin, destination)] = origin_destination.get((origin, destination), 0) + 1
            destination_counts[destination] = destination_counts.get(destination, 0) + 1
            user_trip_counts[user] = user_trip_counts.get(user, 0) + 1
            user_origin_counts[(user, origin)] = user_origin_counts.get((user, origin), 0) + 1
            origin_counts[origin] = origin_counts.get(origin, 0) + 1
    
    df = pd.DataFrame(records)
    df = add_seasonal_features(df, seasonal_features)
    return df

# ============================================================
# MAIN
# ============================================================
print("Loading data...")
global_train = pd.read_csv(DATA_DIR / "journeys_train.csv")
user_train = pd.read_csv(DATA_DIR / "user_train.csv")

for df in [global_train, user_train]:
    df["userCode"] = df["userCode"].astype(str)
    df["origin"] = df["origin"].astype(str)
    df["destination"] = df["destination"].astype(str)
    df["departure_date"] = pd.to_datetime(df["departure_date"])

global_train = global_train.sort_values(["departure_date", "travelCode"]).reset_index(drop=True)
user_train = user_train.sort_values(["departure_date", "travelCode"]).reset_index(drop=True)

global_candidates = sorted(global_train["destination"].unique().tolist())
user_candidates = sorted(user_train["destination"].unique().tolist())

# Build seasonal features
print("Building seasonal features...")
global_seasonal = build_seasonal_features(global_train)
user_seasonal = build_seasonal_features(user_train)

# Preprocessor
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), DEST_CATEGORICAL),
    ("numeric", SimpleImputer(strategy="median"), DEST_NUMERIC),
])

# ============================================================
# TRAIN DESTINATION MODELS
# ============================================================
print("\n=== TRAINING DESTINATION MODELS ===")

for split_name, train_df, candidates, seasonal in [
    ("global_80_20", global_train, global_candidates, global_seasonal),
    ("user_80_20", user_train, user_candidates, user_seasonal),
]:
    print(f"\n--- {split_name} ---")
    
    # Save seasonal features
    joblib.dump(seasonal, MODEL_DIR / f"seasonal_features_{split_name}.joblib")
    
    print(f"Generating training candidates for {split_name}...")
    train_candidates = make_training_candidates(train_df, candidates, seasonal)
    print(f"  Training rows: {len(train_candidates)}")
    
    # Save candidates
    train_candidates.to_csv(MODEL_DIR / f"train_candidates_{split_name}.csv", index=False)
    
    for model_name in ["random_forest", "gradient_boosting", "logistic_regression"]:
        print(f"  Training {model_name}...")
        if model_name == "random_forest":
            clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)
        elif model_name == "gradient_boosting":
            clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, min_samples_leaf=10, random_state=RANDOM_STATE)
        else:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE, solver="lbfgs")
        
        model = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        model.fit(train_candidates[DEST_FEATURES], train_candidates["label"])
        
        # Save model
        model_path = MODEL_DIR / f"dest_model_{model_name}_{split_name}.joblib"
        joblib.dump(model, model_path)
        print(f"    Saved to {model_path}")

# Save candidates
joblib.dump({"global_80_20": global_candidates, "user_80_20": user_candidates}, MODEL_DIR / "dest_candidates.joblib")

# ============================================================
# TRAIN HOTEL MODELS
# ============================================================
print("\n=== TRAINING HOTEL MODELS ===")

hotel_train = pd.read_csv(MODEL_DIR / "hotel_train.csv")
hotel_test = pd.read_csv(MODEL_DIR / "hotel_test.csv")

if "userCode_x" in hotel_train.columns:
    hotel_train["userCode"] = hotel_train["userCode_x"]
    hotel_test["userCode"] = hotel_test["userCode_x"]

X_train = hotel_train[HOTEL_FEATURES].copy()
y_train = hotel_train["hotel_total"].copy()
X_test = hotel_test[HOTEL_FEATURES].copy()
y_test = hotel_test["hotel_total"].copy()

for df in [X_train, X_test]:
    df[HOTEL_NUMERIC] = df[HOTEL_NUMERIC].fillna(0)
    df[HOTEL_CATEGORICAL] = df[HOTEL_CATEGORICAL].fillna("unknown")

hotel_preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), HOTEL_CATEGORICAL),
    ("numeric", StandardScaler(), HOTEL_NUMERIC),
])

for model_name in ["linear_regression", "random_forest", "gradient_boosting"]:
    print(f"  Training hotel {model_name}...")
    if model_name == "linear_regression":
        reg = LinearRegression()
    elif model_name == "random_forest":
        reg = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)
    else:
        reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, min_samples_leaf=10, random_state=RANDOM_STATE)
    
    model = Pipeline([("preprocessor", hotel_preprocessor), ("regressor", reg)])
    model.fit(X_train, y_train)
    
    model_path = MODEL_DIR / f"hotel_model_{model_name}.joblib"
    joblib.dump(model, model_path)
    print(f"    Saved to {model_path}")

print("\n=== ALL MODELS TRAINED AND SAVED ===")