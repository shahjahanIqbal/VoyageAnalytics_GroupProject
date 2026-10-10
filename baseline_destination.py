
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("travel_recommender_outputs")
TOP_K = 5

# --------------------------------------------------
# 1. Load the prepared journey datasets
# --------------------------------------------------
train = pd.read_csv(OUT / "journeys_train.csv")
test = pd.read_csv(OUT / "journeys_test.csv")

print("Training journeys:", len(train))
print("Testing journeys:", len(test))

if train.empty or test.empty:
    raise ValueError("Training or test dataset is empty.")

# --------------------------------------------------
# 2. Learn destination popularity from TRAINING only
# --------------------------------------------------
destination_counts = train["destination"].value_counts()

# Rank destinations from most to least popular.
top_destinations = destination_counts.head(TOP_K).index.tolist()

print("\n=== TOP DESTINATIONS IN TRAINING DATA ===")
print(destination_counts.head(TOP_K))

print("\nBaseline recommendations:")
for rank, destination in enumerate(top_destinations, start=1):
    print(f"{rank}. {destination}")

# --------------------------------------------------
# 3. Evaluate the same ranking for every test journey
# --------------------------------------------------
results = []

for row in test.itertuples(index=False):
    actual = str(row.destination)

    # Global popularity makes the same recommendation
    # regardless of user, origin, or travel date.
    ranked = [str(d) for d in top_destinations]

    if actual in ranked:
        rank = ranked.index(actual) + 1
        reciprocal_rank = 1.0 / rank
    else:
        rank = None
        reciprocal_rank = 0.0

    results.append({
        "travelCode": row.travelCode,
        "userCode": row.userCode,
        "origin": row.origin,
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
# 4. Calculate ranking metrics
# --------------------------------------------------
metrics = {
    "model": "global_popularity",
    "train_journeys": len(train),
    "test_journeys": len(test),
    "unique_test_destinations": test["destination"].nunique(),
    "top1_accuracy": predictions["top1_correct"].mean(),
    "hit_rate_at_5": predictions["hit_at_5"].mean(),
    "mrr_at_5": predictions["reciprocal_rank_at_5"].mean(),
    "catalog_coverage_at_5": (
        len(set(top_destinations)) /
        train["destination"].nunique()
    ),
}

metrics_df = pd.DataFrame([metrics])

print("\n=== BASELINE METRICS ===")
print(metrics_df.to_string(index=False))

# --------------------------------------------------
# 5. Inspect which destinations were missed
# --------------------------------------------------
print("\n=== MOST COMMON TEST DESTINATIONS ===")
print(test["destination"].value_counts().head(10))

print("\n=== TEST DESTINATIONS NOT IN TOP 5 ===")
missed = predictions.loc[
    predictions["hit_at_5"] == 0,
    "actual_destination"
].value_counts()

print(missed.head(15))

# --------------------------------------------------
# 6. Save results
# --------------------------------------------------
metrics_df.to_csv(OUT / "destination_baseline_metrics.csv", index=False)
predictions.to_csv(OUT / "destination_baseline_predictions.csv", index=False)

print(f"\nResults saved to: {OUT.resolve()}")
