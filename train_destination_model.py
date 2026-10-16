
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

OUT = Path("travel_recommender_outputs")
TOP_K = 5
RANDOM_STATE = 42

# --------------------------------------------------
# 1. Load the prepared journey datasets
# --------------------------------------------------
train = pd.read_csv(OUT / "journeys_train.csv")
test = pd.read_csv(OUT / "journeys_test.csv")

for df in (train, test):
    df["userCode"] = df["userCode"].astype(str)
    df["origin"] = df["origin"].astype(str)
    df["destination"] = df["destination"].astype(str)
    df["departure_date"] = pd.to_datetime(df["departure_date"])

train = train.sort_values(["departure_date", "travelCode"]).copy()
test = test.sort_values(["departure_date", "travelCode"]).copy()

# --------------------------------------------------
# 2. Create historical features without target leakage
# --------------------------------------------------
def add_training_history_features(df):
    """For training rows, use only earlier training journeys."""
    df = df.copy()

    df["month"] = df["departure_date"].dt.month
    df["day_of_week"] = df["departure_date"].dt.dayofweek
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    user_destination = {}
    user_route = {}
    origin_destination = {}

    user_counts = []
    route_counts = []
    origin_destination_counts = []

    for row in df[
        ["userCode", "origin", "destination"]
    ].itertuples(index=False, name=None):
        user, origin, destination = row

        user_counts.append(
            user_destination.get((user, destination), 0)
        )
        route_counts.append(
            user_route.get((user, origin, destination), 0)
        )
        origin_destination_counts.append(
            origin_destination.get((origin, destination), 0)
        )

        # Update only after recording features for this journey.
        user_destination[(user, destination)] = (
            user_destination.get((user, destination), 0) + 1
        )
        user_route[(user, origin, destination)] = (
            user_route.get((user, origin, destination), 0) + 1
        )
        origin_destination[(origin, destination)] = (
            origin_destination.get((origin, destination), 0) + 1
        )

    df["user_destination_count_before"] = user_counts
    df["user_route_count_before"] = route_counts
    df["origin_destination_count_before"] = origin_destination_counts

    return df


def add_test_history_features(train_df, test_df):
    """For test rows, use only the frozen training history."""
    test_df = test_df.copy()

    test_df["month"] = test_df["departure_date"].dt.month
    test_df["day_of_week"] = test_df["departure_date"].dt.dayofweek
    test_df["month_sin"] = np.sin(2 * np.pi * test_df["month"] / 12)
    test_df["month_cos"] = np.cos(2 * np.pi * test_df["month"] / 12)

    user_destination = (
        train_df.groupby(["userCode", "destination"])
        .size()
        .to_dict()
    )
    user_route = (
        train_df.groupby(["userCode", "origin", "destination"])
        .size()
        .to_dict()
    )
    origin_destination = (
        train_df.groupby(["origin", "destination"])
        .size()
        .to_dict()
    )

    test_df["user_destination_count_before"] = [
        user_destination.get((u, d), 0)
        for u, d in zip(test_df["userCode"], test_df["destination"])
    ]

    test_df["user_route_count_before"] = [
        user_route.get((u, o, d), 0)
        for u, o, d in zip(
            test_df["userCode"], test_df["origin"], test_df["destination"]
        )
    ]

    test_df["origin_destination_count_before"] = [
        origin_destination.get((o, d), 0)
        for o, d in zip(test_df["origin"], test_df["destination"])
    ]

    return test_df


# IMPORTANT:
# These functions above currently use the actual destination to look up
# historical counts. That would leak the target into the features.
# Instead, use the corrected target-independent feature builder below.

def build_features(train_df, query_df, update_history):
    """
    Build features without using the query row's destination.
    Training rows use earlier training history.
    Test rows use frozen training history only.
    """
    query_df = query_df.copy()

    query_df["month"] = query_df["departure_date"].dt.month
    query_df["day_of_week"] = query_df["departure_date"].dt.dayofweek
    query_df["month_sin"] = np.sin(2 * np.pi * query_df["month"] / 12)
    query_df["month_cos"] = np.cos(2 * np.pi * query_df["month"] / 12)

    user_counts = {}
    route_counts = {}
    origin_destination_counts = {}

    # For training data, construct historical counts before each row.
    if update_history:
        query_df = query_df.sort_values(
            ["departure_date", "travelCode"]
        ).copy()

        uc, rc, odc = [], [], []

        for row in query_df[
            ["userCode", "origin", "destination"]
        ].itertuples(index=False, name=None):
            user, origin, destination = row

            uc.append(sum(
                count for (u, d), count in user_counts.items()
                if u == user
            ))
            rc.append(route_counts.get((user, origin), 0))
            odc.append(origin_destination_counts.get(origin, 0))

            user_counts[(user, destination)] = (
                user_counts.get((user, destination), 0) + 1
            )
            route_counts[(user, origin)] = (
                route_counts.get((user, origin), 0) + 1
            )
            origin_destination_counts[origin] = (
                origin_destination_counts.get(origin, 0) + 1
            )

        query_df["user_history_count"] = uc
        query_df["user_origin_history_count"] = rc
        query_df["origin_history_count"] = odc

    else:
        # Frozen training statistics. None of these uses the test target.
        user_counts = train_df.groupby("userCode").size().to_dict()
        route_counts = train_df.groupby(
            ["userCode", "origin"]
        ).size().to_dict()
        origin_destination_counts = train_df.groupby("origin").size().to_dict()

        query_df["user_history_count"] = (
            query_df["userCode"].map(user_counts).fillna(0)
        )
        query_df["user_origin_history_count"] = [
            route_counts.get((u, o), 0)
            for u, o in zip(query_df["userCode"], query_df["origin"])
        ]
        query_df["origin_history_count"] = (
            query_df["origin"].map(origin_destination_counts).fillna(0)
        )

    return query_df


# Training features are generated chronologically.
train_f = build_features(train, train, update_history=True)

# Test features use only information from the training period.
test_f = build_features(train, test, update_history=False)

# --------------------------------------------------
# 3. Define features and target
# --------------------------------------------------
CATEGORICAL = ["userCode", "origin"]
NUMERIC = [
    "month_sin",
    "month_cos",
    "day_of_week",
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
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])

# --------------------------------------------------
# 4. Train
# --------------------------------------------------
model.fit(train_f[FEATURES], train_f["destination"])

# --------------------------------------------------
# 5. Generate top-five predictions
# --------------------------------------------------
probabilities = model.predict_proba(test_f[FEATURES])
classes = model.named_steps["classifier"].classes_

top_k_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :TOP_K]

results = []

for i, (_, row) in enumerate(test_f.iterrows()):
    ranked = [str(classes[j]) for j in top_k_indices[i]]
    actual = str(row["destination"])

    rank = ranked.index(actual) + 1 if actual in ranked else None

    results.append({
        "travelCode": row["travelCode"],
        "userCode": row["userCode"],
        "origin": row["origin"],
        "departure_date": row["departure_date"],
        "actual_destination": actual,
        "recommended_destinations": " | ".join(ranked),
        "top1_correct": int(bool(ranked) and ranked[0] == actual),
        "hit_at_5": int(actual in ranked),
        "reciprocal_rank_at_5": 1 / rank if rank else 0.0,
        "actual_rank": rank,
    })

predictions = pd.DataFrame(results)

# --------------------------------------------------
# 6. Evaluate and save
# --------------------------------------------------
metrics = pd.DataFrame([{
    "model": "random_forest",
    "train_journeys": len(train_f),
    "test_journeys": len(test_f),
    "top1_accuracy": accuracy_score(
        test_f["destination"], model.predict(test_f[FEATURES])
    ),
    "hit_rate_at_5": predictions["hit_at_5"].mean(),
    "mrr_at_5": predictions["reciprocal_rank_at_5"].mean(),
}])

print("\n=== RANDOM FOREST METRICS ===")
print(metrics.to_string(index=False))

comparison_path = OUT / "destination_baseline_comparison.csv"
if comparison_path.exists():
    comparison = pd.read_csv(comparison_path)
    comparison = pd.concat([comparison, metrics], ignore_index=True)
else:
    comparison = metrics.copy()

comparison.to_csv(OUT / "destination_model_comparison.csv", index=False)
predictions.to_csv(OUT / "random_forest_destination_predictions.csv", index=False)

print(f"\nSaved results to: {OUT.resolve()}")
