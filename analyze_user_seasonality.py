
from pathlib import Path

import pandas as pd

DATA_PATH = Path("travel_recommender_outputs/journeys_train.csv")
OUT = Path("travel_recommender_outputs")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["userCode"] = df["userCode"].astype(str)
df["departure_date"] = pd.to_datetime(
    df["departure_date"], errors="coerce"
)
df = df.dropna(
    subset=["userCode", "destination", "departure_date"]
)

df["month"] = df["departure_date"].dt.month
df["year"] = df["departure_date"].dt.year

# 1. User-level travel volume and month concentration
user_month = (
    df.groupby(["userCode", "month"])
    .size()
    .rename("trips")
    .reset_index()
)

user_summary = (
    df.groupby("userCode")
    .agg(
        total_trips=("destination", "size"),
        active_months=("month", "nunique"),
        active_years=("year", "nunique"),
    )
)

month_totals = user_month.groupby("userCode")["trips"].sum()
month_shares = user_month.assign(
    share=user_month["trips"]
    / user_month["userCode"].map(month_totals)
)

month_concentration = (
    month_shares.groupby("userCode")["share"]
    .max()
    .rename("largest_month_share")
)

user_summary = user_summary.join(month_concentration)

# 2. Repeated destination in the same calendar month
# across different years
user_destination_month = (
    df.groupby(["userCode", "destination", "month"])
    .agg(
        trips=("destination", "size"),
        years=("year", "nunique"),
    )
    .reset_index()
)

recurring = user_destination_month[
    (user_destination_month["trips"] >= 2)
    & (user_destination_month["years"] >= 2)
].sort_values(
    ["years", "trips"], ascending=False
)

# 3. Export results
user_summary.sort_values(
    ["largest_month_share", "total_trips"],
    ascending=False,
).to_csv(OUT / "user_seasonality_summary.csv")

recurring.to_csv(
    OUT / "recurring_user_destination_month.csv",
    index=False,
)

user_month.to_csv(
    OUT / "user_monthly_trip_counts.csv",
    index=False,
)

print("Users analyzed:", len(user_summary))
print(
    "Users with trips in multiple years:",
    int((user_summary["active_years"] >= 2).sum()),
)
print(
    "Users with recurring destination-month patterns:",
    recurring["userCode"].nunique(),
)
print("\nTop recurring destination-month patterns:")
print(recurring.head(20).to_string(index=False))
