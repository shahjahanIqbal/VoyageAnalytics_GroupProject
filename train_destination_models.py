import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

OUT = Path("model_outputs")
OUT.mkdir(parents=True, exist_ok=True)

TOP_K = 5
RANDOM_STATE = 42

# --------------------------------------------------
# 1. Load both splits
# --------------------------------------------------
split_dir = Path("travel_recommender_outputs")

# Global chronological split
global_train = pd.read_csv(split_dir / "journeys_train.csv")
global_test = pd.read_csv(split_dir / "journeys_test.csv")

# User-specific chronological split
user_train = pd.read_csv(split_dir / "user_train.csv")
user_test = pd.read_csv(split_dir / "user_test.csv")

for df in (global_train, global_test, user_train, user_test):
    df["userCode"] = df["userCode"].astype(str)
    df["origin"] = df["origin"].astype(str)
    df["destination"] = df["destination"].astype(str)
    df["departure_date"] = pd.to_datetime(df["departure_date"])

global_train = global_train.sort_values(["departure_date", "travelCode"]).reset_index(drop=True)
global_test = global_test.sort_values(["departure_date", "travelCode"]).reset_index(drop=True)
user_train = user_train.sort_values(["departure_date", "travelCode"]).reset_index(drop=True)
user_test = user_test.sort_values(["departure_date", "travelCode"]).reset_index(drop=True)

# Candidate destinations from training data
global_candidates = sorted(global_train["destination"].unique().tolist())
user_candidates = sorted(user_train["destination"].unique().tolist())

print("=== DATA LOADED ===")
print(f"Global train: {len(global_train)} journeys, {len(global_candidates)} candidates")
print(f"Global test:  {len(global_test)} journeys")
print(f"User train:   {len(user_train)} journeys, {len(user_candidates)} candidates")
print(f"User test:    {len(user_test)} journeys")

# --------------------------------------------------
# 2. Load seasonal patterns
# --------------------------------------------------
recurring_path = split_dir / "recurring_user_destination_month.csv"
monthly_counts_path = split_dir / "user_monthly_trip_counts.csv"

recurring = pd.read_csv(recurring_path) if recurring_path.exists() else pd.DataFrame()
monthly_counts = pd.read_csv(monthly_counts_path) if monthly_counts_path.exists() else pd.DataFrame()

if not recurring.empty:
    recurring["userCode"] = recurring["userCode"].astype(str)
    recurring["destination"] = recurring["destination"].astype(str)
    print(f"Recurring patterns: {len(recurring)} user-destination-month combinations")

if not monthly_counts.empty:
    monthly_counts["userCode"] = monthly_counts["userCode"].astype(str)
    print(f"Monthly trip counts: {len(monthly_counts)} user-month records")

# --------------------------------------------------
# 3. Build seasonal features
# --------------------------------------------------
def build_seasonal_features(train_df, recurring_df, monthly_counts_df):
    """Create seasonal feature lookup tables from training data only."""
    
    # User-destination-month trip counts from training data
    train_df = train_df.copy()
    train_df["month"] = train_df["departure_date"].dt.month
    train_df["year"] = train_df["departure_date"].dt.year
    
    # Actual seasonal counts from training data
    seasonal_counts = train_df.groupby(["userCode", "destination", "month"]).agg(
        seasonal_trips=("travelCode", "count"),
        seasonal_years=("year", "nunique")
    ).reset_index()
    
    # User's total trips per month
    user_month_totals = train_df.groupby(["userCode", "month"]).agg(
        user_month_total=("travelCode", "count")
    ).reset_index()
    
    # User's total trips per destination
    user_dest_totals = train_df.groupby(["userCode", "destination"]).agg(
        user_dest_total=("travelCode", "count")
    ).reset_index()
    
    # Global destination popularity per month
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
    """Add seasonal features to candidate-level dataframe."""
    df = df.copy()
    
    # Merge seasonal counts
    seasonal = feature_dict["seasonal_counts"].rename(columns={
        "userCode": "userCode",
        "destination": "candidate",
        "month": "month"
    })
    df = df.merge(
        seasonal[["userCode", "candidate", "month", "seasonal_trips", "seasonal_years"]],
        on=["userCode", "candidate", "month"],
        how="left"
    )
    df["seasonal_trips"] = df["seasonal_trips"].fillna(0).astype(int)
    df["seasonal_years"] = df["seasonal_years"].fillna(0).astype(int)
    
    # User month totals
    user_month = feature_dict["user_month_totals"].rename(columns={"userCode": "userCode", "month": "month"})
    df = df.merge(user_month, on=["userCode", "month"], how="left")
    df["user_month_total"] = df["user_month_total"].fillna(0).astype(int)
    
    # User destination totals
    user_dest = feature_dict["user_dest_totals"].rename(columns={"userCode": "userCode", "destination": "candidate"})
    df = df.merge(user_dest, on=["userCode", "candidate"], how="left")
    df["user_dest_total"] = df["user_dest_total"].fillna(0).astype(int)
    
    # Global month-destination rate
    global_md = feature_dict["global_month_dest"][["month", "candidate", "global_month_dest_rate"]].rename(
        columns={"destination": "candidate"}
    )
    df = df.merge(global_md, on=["month", "candidate"], how="left")
    df["global_month_dest_rate"] = df["global_month_dest_rate"].fillna(0)
    
    # Engineered seasonal features
    df["seasonal_strength"] = (df["seasonal_trips"] + smoothing) / (df["user_month_total"] + smoothing)
    df["is_recurring_seasonal"] = ((df["seasonal_trips"] >= 2) & (df["seasonal_years"] >= 2)).astype(int)
    df["user_dest_month_share"] = df["seasonal_trips"] / (df["user_dest_total"] + 1)
    df["dest_month_popularity"] = df["global_month_dest_rate"]
    
    return df

# Build seasonal features for both splits
print("\n=== BUILDING SEASONAL FEATURES ===")
global_seasonal_features = build_seasonal_features(global_train, recurring, monthly_counts)
user_seasonal_features = build_seasonal_features(user_train, recurring, monthly_counts)

# --------------------------------------------------
# 4. Build candidate-level training/test examples
# --------------------------------------------------
def make_training_candidates(journeys, candidate_destinations, seasonal_features):
    """Generate candidate-level training examples with seasonal features."""
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

def make_test_candidates(train_journeys, test_journeys, candidate_destinations, seasonal_features):
    """Generate test candidates using frozen training history."""
    user_destination = train_journeys.groupby(["userCode", "destination"]).size().to_dict()
    user_route = train_journeys.groupby(["userCode", "origin", "destination"]).size().to_dict()
    origin_destination = train_journeys.groupby(["origin", "destination"]).size().to_dict()
    destination_counts = train_journeys["destination"].value_counts().to_dict()
    user_trip_counts = train_journeys["userCode"].value_counts().to_dict()
    user_origin_counts = train_journeys.groupby(["userCode", "origin"]).size().to_dict()
    origin_counts = train_journeys["origin"].value_counts().to_dict()
    
    records = []
    
    for row in test_journeys.itertuples(index=False):
        user = row.userCode
        origin = row.origin
        actual = row.destination
        month = row.departure_date.month
        weekday = row.departure_date.dayofweek
        
        for candidate in candidate_destinations:
            records.append({
                "travelCode": row.travelCode,
                "userCode": user,
                "origin": origin,
                "candidate": candidate,
                "departure_date": row.departure_date,
                "actual_destination": actual,
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
    
    df = pd.DataFrame(records)
    df = add_seasonal_features(df, seasonal_features)
    return df

# Generate candidate datasets for both splits
print("\nGenerating candidate-level training examples...")
global_train_candidates = make_training_candidates(global_train, global_candidates, global_seasonal_features)
global_test_candidates = make_test_candidates(global_train, global_test, global_candidates, global_seasonal_features)

user_train_candidates = make_training_candidates(user_train, user_candidates, user_seasonal_features)
user_test_candidates = make_test_candidates(user_train, user_test, user_candidates, user_seasonal_features)

print(f"Global train candidates: {len(global_train_candidates)}")
print(f"Global test candidates:  {len(global_test_candidates)}")
print(f"User train candidates:   {len(user_train_candidates)}")
print(f"User test candidates:    {len(user_test_candidates)}")

# Sanity checks
assert global_train_candidates.groupby("travelCode")["label"].sum().eq(1).all()
assert global_test_candidates.groupby("travelCode")["label"].sum().eq(1).all()
assert user_train_candidates.groupby("travelCode")["label"].sum().eq(1).all()
assert user_test_candidates.groupby("travelCode")["label"].sum().eq(1).all()

# --------------------------------------------------
# 5. Define features and models
# --------------------------------------------------
CATEGORICAL = ["userCode", "origin", "candidate"]
NUMERIC = [
    "month_sin", "month_cos", "day_of_week",
    "user_candidate_count", "user_route_count", "origin_candidate_count",
    "candidate_popularity", "user_history_count", "user_origin_history_count",
    "origin_history_count",
    "seasonal_trips", "seasonal_years", "user_month_total", "user_dest_total",
    "global_month_dest_rate", "seasonal_strength", "is_recurring_seasonal",
    "user_dest_month_share", "dest_month_popularity",
]

FEATURES = CATEGORICAL + NUMERIC

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ("numeric", SimpleImputer(strategy="median"), NUMERIC),
])

models = {
    "random_forest": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ]),
    "gradient_boosting": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
        )),
    ]),
    "logistic_regression": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            solver="lbfgs",
        )),
    ]),
}

# --------------------------------------------------
# 6. Train and evaluate on both splits
# --------------------------------------------------
def evaluate_model(model, model_name, train_cand, test_cand, candidates, split_name):
    """Train model and evaluate on test candidates."""
    print(f"\n--- Training {model_name} on {split_name} split ---")
    
    model.fit(train_cand[FEATURES], train_cand["label"])
    
    # Score test candidates
    test_cand = test_cand.copy()
    test_cand["score"] = model.predict_proba(test_cand[FEATURES])[:, 1]
    
    test_cand = test_cand.sort_values(["travelCode", "score"], ascending=[True, False])
    
    predictions = []
    for travel_code, group in test_cand.groupby("travelCode", sort=False):
        group = group.sort_values("score", ascending=False)
        top = group.head(TOP_K)
        
        actual = str(group["actual_destination"].iloc[0])
        ranked = top["candidate"].astype(str).tolist()
        
        rank = ranked.index(actual) + 1 if actual in ranked else None
        
        predictions.append({
            "split": split_name,
            "model": model_name,
            "travelCode": travel_code,
            "userCode": group["userCode"].iloc[0],
            "origin": group["origin"].iloc[0],
            "departure_date": group["departure_date"].iloc[0],
            "actual_destination": actual,
            "recommended_destinations": " | ".join(ranked),
            "top1_correct": int(bool(ranked) and ranked[0] == actual),
            "hit_at_5": int(actual in ranked),
            "reciprocal_rank_at_5": 1 / rank if rank else 0.0,
            "actual_rank": rank,
        })
    
    pred_df = pd.DataFrame(predictions)
    
    metrics = {
        "split": split_name,
        "model": model_name,
        "train_journeys": len(train_cand["travelCode"].unique()),
        "test_journeys": len(test_cand["travelCode"].unique()),
        "candidate_destinations": len(candidates),
        "top1_accuracy": pred_df["top1_correct"].mean(),
        "hit_rate_at_5": pred_df["hit_at_5"].mean(),
        "mrr_at_5": pred_df["reciprocal_rank_at_5"].mean(),
    }
    
    print(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
    print(f"  Hit Rate@5:     {metrics['hit_rate_at_5']:.4f}")
    print(f"  MRR@5:          {metrics['mrr_at_5']:.4f}")
    
    return metrics, pred_df

# Run all experiments
all_metrics = []
all_predictions = []

experiments = [
    ("global_80_20", global_train_candidates, global_test_candidates, global_candidates),
    ("user_80_20", user_train_candidates, user_test_candidates, user_candidates),
]

for split_name, train_cand, test_cand, candidates in experiments:
    for model_name, model in models.items():
        metrics, preds = evaluate_model(model, model_name, train_cand, test_cand, candidates, split_name)
        all_metrics.append(metrics)
        all_predictions.append(preds)
        
        # Save individual predictions
        pred_path = OUT / f"predictions_{split_name}_{model_name}.csv"
        preds.to_csv(pred_path, index=False)

# --------------------------------------------------
# 7. Save results and select best model
# --------------------------------------------------
metrics_df = pd.DataFrame(all_metrics)
metrics_path = OUT / "destination_model_comparison.csv"
metrics_df.to_csv(metrics_path, index=False)

predictions_df = pd.concat(all_predictions, ignore_index=True)
predictions_path = OUT / "destination_all_predictions.csv"
predictions_df.to_csv(predictions_path, index=False)

# Select best model by Hit@5 (primary) then MRR@5 (secondary)
best_idx = metrics_df.sort_values(["hit_rate_at_5", "mrr_at_5"], ascending=[False, False]).index[0]
best_model_info = metrics_df.loc[best_idx]
best_split = best_model_info["split"]
best_model_name = best_model_info["model"]

print("\n=== BEST MODEL ===")
print(f"Split: {best_split}")
print(f"Model: {best_model_name}")
print(f"Hit@5: {best_model_info['hit_rate_at_5']:.4f}")
print(f"MRR@5: {best_model_info['mrr_at_5']:.4f}")

# Retrain best model on full training data for that split
if best_split == "global_80_20":
    best_train_cand = global_train_candidates
    best_candidates = global_candidates
else:
    best_train_cand = user_train_candidates
    best_candidates = user_candidates

best_model = models[best_model_name]
print(f"\nRetraining best model ({best_model_name}) on full {best_split} training data...")
best_model.fit(best_train_cand[FEATURES], best_train_cand["label"])

# Save best model
model_path = OUT / "best_destination_model.joblib"
joblib.dump({
    "model": best_model,
    "features": FEATURES,
    "candidates": best_candidates,
    "split": best_split,
    "model_name": best_model_name,
    "metrics": best_model_info.to_dict(),
}, model_path)

print(f"\nBest model saved to: {model_path.resolve()}")
print(f"Metrics saved to: {metrics_path.resolve()}")
print(f"All predictions saved to: {predictions_path.resolve()}")

# Print summary table
print("\n=== FINAL COMPARISON ===")
print(metrics_df[["split", "model", "top1_accuracy", "hit_rate_at_5", "mrr_at_5"]]
      .sort_values(["split", "hit_rate_at_5"], ascending=[True, False])
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))