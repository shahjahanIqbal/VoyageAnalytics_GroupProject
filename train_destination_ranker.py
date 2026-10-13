
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

OUT = Path("travel_recommender_outputs")
TOP_K = 5
RANDOM_STATE = 42

# --------------------------------------------------
# 1. Load the existing chronological split
# --------------------------------------------------
train = pd.read_csv(OUT / "journeys_train.csv")
test = pd.read_csv(OUT / "journeys_test.csv")

for df in (train, test):
    df["userCode"] = df["userCode"].astype(str)
    df["origin"] = df["origin"].astype(str)
    df["destination"] = df["destination"].astype(str)
    df["departure_date"] = pd.to_datetime(df["departure_date"])

train = train.sort_values(["departure_date", "travelCode"]).reset_index(drop=True)
test = test.sort_values(["departure_date", "travelCode"]).reset_index(drop=True)

if train.empty or test.empty:
    raise ValueError("Training or test data is empty.")

# Only recommend destinations represented in training.
candidates = sorted(train["destination"].unique().tolist())

print("Training journeys:", len(train))
print("Test journeys:", len(test))
print("Candidate destinations:", len(candidates))

# --------------------------------------------------
# 2. Build candidate-level training examples
# --------------------------------------------------
def make_training_candidates(journeys, candidate_destinations):
    """
    For each journey, generate one row per candidate destination.

    History features are computed from journeys strictly earlier
    than the current departure date. All journeys on the same date
    receive the same pre-date history.
    """
    journeys = journeys.sort_values(
        ["departure_date", "travelCode"]
    ).copy()

    user_destination = {}
    user_route = {}
    origin_destination = {}
    destination_counts = {}
    user_trip_counts = {}
    user_origin_counts = {}
    origin_counts = {}

    records = []

    for date, daily in journeys.groupby("departure_date", sort=True):
        # Build examples using history before this date.
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
                    "user_candidate_count": user_destination.get(
                        (user, candidate), 0
                    ),
                    "user_route_count": user_route.get(
                        (user, origin, candidate), 0
                    ),
                    "origin_candidate_count": origin_destination.get(
                        (origin, candidate), 0
                    ),
                    "candidate_popularity": destination_counts.get(
                        candidate, 0
                    ),
                    "user_history_count": user_trip_counts.get(user, 0),
                    "user_origin_history_count": user_origin_counts.get(
                        (user, origin), 0
                    ),
                    "origin_history_count": origin_counts.get(origin, 0),
                    "label": int(candidate == actual),
                })

        # Update histories only after processing every journey on this date.
        for row in daily.itertuples(index=False):
            user = row.userCode
            origin = row.origin
            destination = row.destination

            user_destination[(user, destination)] = (
                user_destination.get((user, destination), 0) + 1
            )
            user_route[(user, origin, destination)] = (
                user_route.get((user, origin, destination), 0) + 1
            )
            origin_destination[(origin, destination)] = (
                origin_destination.get((origin, destination), 0) + 1
            )
            destination_counts[destination] = (
                destination_counts.get(destination, 0) + 1
            )
            user_trip_counts[user] = user_trip_counts.get(user, 0) + 1
            user_origin_counts[(user, origin)] = (
                user_origin_counts.get((user, origin), 0) + 1
            )
            origin_counts[origin] = origin_counts.get(origin, 0) + 1

    return pd.DataFrame(records)


# --------------------------------------------------
# 3. Build test examples using frozen training history
# --------------------------------------------------
def make_test_candidates(train_journeys, test_journeys, candidate_destinations):
    """
    Each test prediction uses the full training history.
    Test labels are used only for evaluation, never as features.
    """
    user_destination = (
        train_journeys.groupby(["userCode", "destination"])
        .size().to_dict()
    )
    user_route = (
        train_journeys.groupby(["userCode", "origin", "destination"])
        .size().to_dict()
    )
    origin_destination = (
        train_journeys.groupby(["origin", "destination"])
        .size().to_dict()
    )
    destination_counts = train_journeys["destination"].value_counts().to_dict()
    user_trip_counts = train_journeys["userCode"].value_counts().to_dict()
    user_origin_counts = (
        train_journeys.groupby(["userCode", "origin"])
        .size().to_dict()
    )
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
                "user_candidate_count": user_destination.get(
                    (user, candidate), 0
                ),
                "user_route_count": user_route.get(
                    (user, origin, candidate), 0
                ),
                "origin_candidate_count": origin_destination.get(
                    (origin, candidate), 0
                ),
                "candidate_popularity": destination_counts.get(
                    candidate, 0
                ),
                "user_history_count": user_trip_counts.get(user, 0),
                "user_origin_history_count": user_origin_counts.get(
                    (user, origin), 0
                ),
                "origin_history_count": origin_counts.get(origin, 0),
                "label": int(candidate == actual),
            })

    return pd.DataFrame(records)


print("\nGenerating candidate-level training examples...")
train_candidates = make_training_candidates(train, candidates)

print("Training candidate rows:", len(train_candidates))
print("Positive labels:", int(train_candidates["label"].sum()))
print("Expected positive labels:", len(train))

print("\nGenerating test candidates...")
test_candidates = make_test_candidates(train, test, candidates)

# Sanity checks: each journey should have exactly one positive candidate.
assert train_candidates.groupby("travelCode")["label"].sum().eq(1).all()
assert test_candidates.groupby("travelCode")["label"].sum().eq(1).all()

# --------------------------------------------------
# 4. Define features and train the ranker
# --------------------------------------------------
CATEGORICAL = ["userCode", "origin", "candidate"]

NUMERIC = [
    "month_sin",
    "month_cos",
    "day_of_week",
    "user_candidate_count",
    "user_route_count",
    "origin_candidate_count",
    "candidate_popularity",
    "user_history_count",
    "user_origin_history_count",
    "origin_history_count",
]

FEATURES = CATEGORICAL + NUMERIC

preprocessor = ColumnTransformer([
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        CATEGORICAL,
    ),
    (
        "numeric",
        SimpleImputer(strategy="median"),
        NUMERIC,
    ),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])

print("\nTraining candidate ranker...")
model.fit(
    train_candidates[FEATURES],
    train_candidates["label"],
)

# Save the fitted model for inference and MLflow artifact tracking.
model_path = OUT / "candidate_ranker.joblib"
joblib.dump(model, model_path)
print(f"Saved Random Forest pipeline to: {model_path.resolve()}")

# --------------------------------------------------
# 5. Rank destinations for each test journey
# --------------------------------------------------
print("Scoring test candidates...")
test_candidates["score"] = model.predict_proba(
    test_candidates[FEATURES]
)[:, list(model.named_steps["classifier"].classes_).index(1)]

test_candidates = test_candidates.sort_values(
    ["travelCode", "score"],
    ascending=[True, False],
)

predictions = []

for travel_code, group in test_candidates.groupby("travelCode", sort=False):
    group = group.sort_values("score", ascending=False)
    top = group.head(TOP_K)

    actual = str(group["actual_destination"].iloc[0])
    ranked = top["candidate"].astype(str).tolist()

    rank = ranked.index(actual) + 1 if actual in ranked else None

    predictions.append({
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

predictions = pd.DataFrame(predictions)

# --------------------------------------------------
# 6. Evaluate and save
# --------------------------------------------------
metrics = pd.DataFrame([{
    "model": "candidate_random_forest",
    "train_journeys": len(train),
    "test_journeys": len(test),
    "candidate_destinations": len(candidates),
    "top1_accuracy": predictions["top1_correct"].mean(),
    "hit_rate_at_5": predictions["hit_at_5"].mean(),
    "mrr_at_5": predictions["reciprocal_rank_at_5"].mean(),
}])

print("\n=== CANDIDATE RANKER METRICS ===")
print(metrics.to_string(index=False))

metrics.to_csv(OUT / "candidate_ranker_metrics.csv", index=False)
predictions.to_csv(
    OUT / "candidate_ranker_predictions.csv",
    index=False,
)

print(f"\nSaved outputs to: {OUT.resolve()}")



# ============================================================
# SEASONAL DESTINATION RECOMMENDER
# ============================================================

SEASONAL_WEIGHTS = {
    "user_destination": 0.45,
    "user_destination_month": 0.35,
    "user_route": 0.15,
    "global_destination": 0.05,
}

SEASONAL_SMOOTHING = 5.0


def recommend_destinations_seasonal(
    user,
    origin,
    departure_month,
    history,
    k=TOP_K,
):
    """
    Rank destinations using only the supplied historical data.

    history must contain only journeys available before the
    recommendation date. It must include userCode, from, to,
    and date columns.
    """
    history = history.copy()

    if history.empty:
        return []

    history["userCode"] = history["userCode"].astype(str)
    history["from"] = history["from"].astype(str)
    history["to"] = history["to"].astype(str)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")

    history = history.dropna(subset=["date", "userCode", "from", "to"])

    user = str(user)
    origin = str(origin)
    departure_month = int(departure_month)

    destinations = sorted(history["to"].unique())

    if not destinations:
        return []

    user_history = history[history["userCode"] == user]
    user_destination_history = user_history[
        user_history["to"].isin(destinations)
    ]
    user_month_history = user_destination_history[
        user_destination_history["date"].dt.month == departure_month
    ]
    user_route_history = user_history[
        user_history["from"] == origin
    ]

    global_counts = history["to"].value_counts()
    global_total = global_counts.sum()

    user_counts = user_destination_history["to"].value_counts()
    user_total = user_counts.sum()

    seasonal_counts = user_month_history["to"].value_counts()
    seasonal_total = seasonal_counts.sum()

    route_counts = user_route_history["to"].value_counts()
    route_total = route_counts.sum()

    # Smooth sparse seasonal history toward the user's overall
    # destination preferences instead of overreacting to one trip.
    user_prior = {
        destination: (
            user_counts.get(destination, 0) / user_total
            if user_total else 0.0
        )
        for destination in destinations
    }

    seasonal_scores = {}
    for destination in destinations:
        seasonal_count = seasonal_counts.get(destination, 0)
        seasonal_scores[destination] = (
            seasonal_count
            + SEASONAL_SMOOTHING * user_prior[destination]
        ) / (
            seasonal_total + SEASONAL_SMOOTHING
        )

    # Use uniform fallback distributions when a user has no history.
    global_scores = {
        destination: (
            global_counts.get(destination, 0) / global_total
            if global_total else 1 / len(destinations)
        )
        for destination in destinations
    }

    route_scores = {
        destination: (
            route_counts.get(destination, 0) / route_total
            if route_total else user_prior[destination]
        )
        for destination in destinations
    }

    combined_scores = {}

    for destination in destinations:
        combined_scores[destination] = (
            SEASONAL_WEIGHTS["user_destination"]
            * user_prior[destination]
            + SEASONAL_WEIGHTS["user_destination_month"]
            * seasonal_scores[destination]
            + SEASONAL_WEIGHTS["user_route"]
            * route_scores[destination]
            + SEASONAL_WEIGHTS["global_destination"]
            * global_scores[destination]
        )

    ranked = sorted(
        combined_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return [(destination, float(score)) for destination, score in ranked[:k]]



def evaluate_seasonal_destination(test, history, split_name):
    """
    Evaluate seasonal recommendations using historical journeys
    strictly earlier than each test journey's date.
    """
    results = []

    history = history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    test = test.copy()
    test["date"] = pd.to_datetime(test["date"], errors="coerce")

    for _, row in test.sort_values("date").iterrows():
        user = str(row["userCode"])
        origin = str(row["from"])
        actual = str(row["to"])
        target_date = row["date"]

        # Exclude the target journey and all future journeys.
        past = history[history["date"] < target_date]

        ranked = recommend_destinations_seasonal(
            user=user,
            origin=origin,
            departure_month=target_date.month,
            history=past,
            k=TOP_K,
        )

        recommended = [destination for destination, _ in ranked]

        rank = (
            recommended.index(actual) + 1
            if actual in recommended else None
        )

        results.append({
            "split": split_name,
            "userCode": user,
            "date": target_date,
            "actual_destination": actual,
            "recommended_destinations": recommended,
            "top1_correct": bool(recommended and recommended[0] == actual),
            "hit_at_5": int(actual in recommended),
            "reciprocal_rank": 1 / rank if rank is not None else 0.0,
            "historical_rows": len(past),
        })

    predictions = pd.DataFrame(results)

    metrics = {
        "split": split_name,
        "stage": "destination_seasonal",
        "test_rows": len(predictions),
        "top1_accuracy": predictions["top1_correct"].mean(),
        "hit_rate_at_5": predictions["hit_at_5"].mean(),
        "mrr_at_5": predictions["reciprocal_rank"].mean(),
    }

    return predictions, metrics



# ============================================================
# 7. SEASONAL DESTINATION RECOMMENDER
# ============================================================

SEASONAL_WEIGHTS = {
    "user_destination": 0.45,
    "user_destination_month": 0.35,
    "user_route": 0.15,
    "global_destination": 0.05,
}

SEASONAL_SMOOTHING = 5.0


def recommend_destinations_seasonal(
    user,
    origin,
    departure_month,
    history,
    candidates,
    k=TOP_K,
):
    """Rank destinations using historical journeys only."""

    user = str(user)
    origin = str(origin)
    candidates = sorted(map(str, candidates))

    if not candidates:
        return []

    history = history.copy()

    history["userCode"] = history["userCode"].astype(str)
    history["origin"] = history["origin"].astype(str)
    history["destination"] = history["destination"].astype(str)
    history["departure_date"] = pd.to_datetime(
        history["departure_date"], errors="coerce"
    )

    history = history.dropna(subset=["departure_date"])

    user_history = history[history["userCode"] == user]

    # 1. Overall destination preferences for this user.
    user_counts = user_history["destination"].value_counts()
    user_total = len(user_history)

    user_prior = {
        d: user_counts.get(d, 0) / user_total
        if user_total else 0.0
        for d in candidates
    }

    # 2. Destinations this user visited in the target month.
    monthly_history = user_history[
        user_history["departure_date"].dt.month == departure_month
    ]
    monthly_counts = monthly_history["destination"].value_counts()
    monthly_total = len(monthly_history)

    # Smooth monthly preferences toward overall user preferences.
    seasonal_scores = {
        d: (
            monthly_counts.get(d, 0)
            + SEASONAL_SMOOTHING * user_prior[d]
        ) / (
            monthly_total + SEASONAL_SMOOTHING
        )
        for d in candidates
    }

    # 3. This user's destination preferences from this origin.
    route_history = user_history[
        user_history["origin"] == origin
    ]
    route_counts = route_history["destination"].value_counts()
    route_total = len(route_history)

    route_scores = {
        d: route_counts.get(d, 0) / route_total
        if route_total else user_prior[d]
        for d in candidates
    }

    # 4. Global destination popularity.
    global_counts = history["destination"].value_counts()
    global_total = len(history)

    global_scores = {
        d: global_counts.get(d, 0) / global_total
        if global_total else 1 / len(candidates)
        for d in candidates
    }

    # Combine the four signals.
    scores = {}

    for d in candidates:
        scores[d] = (
            SEASONAL_WEIGHTS["user_destination"] * user_prior[d]
            + SEASONAL_WEIGHTS["user_destination_month"]
            * seasonal_scores[d]
            + SEASONAL_WEIGHTS["user_route"] * route_scores[d]
            + SEASONAL_WEIGHTS["global_destination"] * global_scores[d]
        )

    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return ranked[:k]


def evaluate_seasonal_destination(train, test, candidates):
    """Evaluate using the same frozen training history as the ranker."""

    results = []

    for row in test.itertuples(index=False):
        ranked = recommend_destinations_seasonal(
            user=row.userCode,
            origin=row.origin,
            departure_month=row.departure_date.month,
            history=train,
            candidates=candidates,
        )

        recommended = [destination for destination, _ in ranked]
        actual = str(row.destination)

        rank = (
            recommended.index(actual) + 1
            if actual in recommended else None
        )

        results.append({
            "travelCode": row.travelCode,
            "userCode": row.userCode,
            "origin": row.origin,
            "departure_date": row.departure_date,
            "actual_destination": actual,
            "recommended_destinations": " | ".join(recommended),
            "top1_correct": int(
                bool(recommended) and recommended[0] == actual
            ),
            "hit_at_5": int(actual in recommended),
            "reciprocal_rank_at_5": 1 / rank if rank else 0.0,
            "actual_rank": rank,
        })

    predictions = pd.DataFrame(results)

    metrics = pd.DataFrame([{
        "model": "seasonal_personalized",
        "train_journeys": len(train),
        "test_journeys": len(test),
        "candidate_destinations": len(candidates),
        "top1_accuracy": predictions["top1_correct"].mean(),
        "hit_rate_at_5": predictions["hit_at_5"].mean(),
        "mrr_at_5": predictions["reciprocal_rank_at_5"].mean(),
    }])

    return predictions, metrics


# ------------------------------------------------------------
# 8. Evaluate seasonal recommender and compare
# ------------------------------------------------------------

seasonal_predictions, seasonal_metrics = evaluate_seasonal_destination(
    train=train,
    test=test,
    candidates=candidates,
)

seasonal_predictions.to_csv(
    OUT / "seasonal_destination_predictions.csv",
    index=False,
)

seasonal_metrics.to_csv(
    OUT / "seasonal_destination_metrics.csv",
    index=False,
)

comparison = pd.concat(
    [metrics, seasonal_metrics],
    ignore_index=True,
)

comparison.to_csv(
    OUT / "destination_ranker_seasonal_comparison.csv",
    index=False,
)

print("\n=== RANDOM FOREST VS SEASONAL RECOMMENDER ===")
print(
    comparison[
        [
            "model",
            "top1_accuracy",
            "hit_rate_at_5",
            "mrr_at_5",
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

print(f"\nSeasonal results saved to: {OUT.resolve()}")
