
import pandas as pd
from pathlib import Path

OUT = Path("travel_recommender_outputs")
TOP_K = 5

train = pd.read_csv(OUT / "journeys_train.csv")
test = pd.read_csv(OUT / "journeys_test.csv")

for df in (train, test):
    df["userCode"] = df["userCode"].astype(str)
    df["origin"] = df["origin"].astype(str)
    df["destination"] = df["destination"].astype(str)

train["departure_date"] = pd.to_datetime(train["departure_date"])
test["departure_date"] = pd.to_datetime(test["departure_date"])

if train.empty or test.empty:
    raise ValueError("Training or test dataset is empty.")

# --------------------------------------------------
# 1. Build history from training data only
# --------------------------------------------------

# Global destination popularity.
global_counts = train["destination"].value_counts()

# Destinations previously visited by each user.
user_destination_counts = (
    train.groupby(["userCode", "destination"])
    .size()
    .to_dict()
)

# Destinations previously visited from each origin.
origin_destination_counts = (
    train.groupby(["origin", "destination"])
    .size()
    .to_dict()
)

# Destinations visited by each user from each origin.
user_route_counts = (
    train.groupby(["userCode", "origin", "destination"])
    .size()
    .to_dict()
)

# --------------------------------------------------
# 2. Rank candidates for each test journey
# --------------------------------------------------

results = []

for row in test.itertuples(index=False):
    user = row.userCode
    origin = row.origin
    actual = row.destination

    # Candidate destinations are those observed in training.
    candidates = set(global_counts.index.astype(str))

    ranked_scores = []

    for destination in candidates:
        route_count = user_route_counts.get(
            (user, origin, destination), 0
        )
        user_count = user_destination_counts.get(
            (user, destination), 0
        )
        origin_count = origin_destination_counts.get(
            (origin, destination), 0
        )
        global_count = global_counts.get(destination, 0)

        # Lexicographic priority:
        # 1. User's history on this origin-destination route
        # 2. User's destination history
        # 3. Route popularity from this origin
        # 4. Global destination popularity
        score = (
            route_count,
            user_count,
            origin_count,
            global_count,
        )

        ranked_scores.append((destination, score))

    ranked_scores.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    ranked = [d for d, _ in ranked_scores[:TOP_K]]

    if actual in ranked:
        rank = ranked.index(actual) + 1
        reciprocal_rank = 1.0 / rank
    else:
        rank = None
        reciprocal_rank = 0.0

    results.append({
        "travelCode": row.travelCode,
        "userCode": user,
        "origin": origin,
        "departure_date": row.departure_date,
        "actual_destination": actual,
        "recommended_destinations": " | ".join(ranked),
        "top1_correct": int(bool(ranked) and ranked[0] == actual),
        "hit_at_5": int(actual in ranked),
        "reciprocal_rank_at_5": reciprocal_rank,
        "actual_rank": rank,
    })

predictions = pd.DataFrame(results)

# --------------------------------------------------
# 3. Evaluate
# --------------------------------------------------

metrics = {
    "model": "personalized_history_baseline",
    "train_journeys": len(train),
    "test_journeys": len(test),
    "unique_test_destinations": test["destination"].nunique(),
    "top1_accuracy": predictions["top1_correct"].mean(),
    "hit_rate_at_5": predictions["hit_at_5"].mean(),
    "mrr_at_5": predictions["reciprocal_rank_at_5"].mean(),
}

metrics_df = pd.DataFrame([metrics])

print("\n=== PERSONALIZED BASELINE METRICS ===")
print(metrics_df.to_string(index=False))

# --------------------------------------------------
# 4. Compare against the global popularity baseline
# --------------------------------------------------

global_path = OUT / "destination_baseline_metrics.csv"

if global_path.exists():
    global_metrics = pd.read_csv(global_path)

    comparison = pd.concat(
        [global_metrics, metrics_df],
        ignore_index=True,
    )

    print("\n=== BASELINE COMPARISON ===")
    print(
        comparison[
            [
                "model",
                "top1_accuracy",
                "hit_rate_at_5",
                "mrr_at_5",
            ]
        ].to_string(index=False)
    )

    comparison.to_csv(
        OUT / "destination_baseline_comparison.csv",
        index=False,
    )

predictions.to_csv(
    OUT / "personalized_destination_predictions.csv",
    index=False,
)

metrics_df.to_csv(
    OUT / "personalized_destination_metrics.csv",
    index=False,
)

print(f"\nResults saved to: {OUT.resolve()}")
