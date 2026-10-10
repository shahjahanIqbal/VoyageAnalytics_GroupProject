
import pandas as pd
from pathlib import Path

DATA_DIR = Path(".")
OUT = Path("travel_recommender_outputs")
OUT.mkdir(exist_ok=True)

# --------------------------------------------------
# 1. Load and validate the datasets
# --------------------------------------------------
flights = pd.read_csv(DATA_DIR / "flights.csv")
hotels = pd.read_csv(DATA_DIR / "hotels.csv")
users = pd.read_csv(DATA_DIR / "users.csv")

for df in (flights, hotels, users):
    df.columns = df.columns.str.strip()

flights["date"] = pd.to_datetime(flights["date"], errors="coerce")
hotels["date"] = pd.to_datetime(hotels["date"], errors="coerce")

flights = flights.dropna(
    subset=["travelCode", "userCode", "from", "to", "date"]
).copy()

hotels = hotels.dropna(
    subset=["travelCode", "userCode", "name", "place", "date"]
).copy()

flights["userCode"] = flights["userCode"].astype(str)
hotels["userCode"] = hotels["userCode"].astype(str)

# --------------------------------------------------
# 2. Reconstruct one row per journey
# --------------------------------------------------
flights = flights.sort_values(["travelCode", "date"])

journeys = (
    flights.groupby("travelCode", as_index=False)
    .agg(
        userCode=("userCode", "first"),
        origin=("from", "first"),
        destination=("to", "first"),
        departure_date=("date", "first"),
        return_origin=("from", "last"),
        return_destination=("to", "last"),
        return_date=("date", "last"),
        flight_rows=("travelCode", "size"),
        user_count=("userCode", "nunique"),
    )
)

# Verify the expected journey structure.
assert journeys["travelCode"].is_unique
assert journeys["user_count"].eq(1).all()
assert journeys["flight_rows"].eq(2).all()
assert (
    journeys["origin"] == journeys["return_destination"]
).all()
assert (
    journeys["destination"] == journeys["return_origin"]
).all()

journeys = journeys.drop(columns=["user_count"])

# --------------------------------------------------
# 3. Join hotel records
# --------------------------------------------------
hotels = hotels.rename(columns={
    "name": "hotelName",
    "place": "hotel_place",
    "date": "hotel_date",
})

# Keep one canonical userCode column and verify the join.
hotel_user_check = hotels[["travelCode", "userCode"]].rename(
    columns={"userCode": "hotel_userCode"}
)

journeys = journeys.merge(
    hotel_user_check,
    on="travelCode",
    how="left",
    validate="one_to_one",
)

matched = journeys["hotel_userCode"].notna()
assert (
    journeys.loc[matched, "userCode"]
    == journeys.loc[matched, "hotel_userCode"]
).all()

journeys = journeys.drop(columns=["hotel_userCode"])

journeys = journeys.merge(
    hotels.drop(columns=["userCode"]),
    on="travelCode",
    how="left",
    validate="one_to_one",
)

has_hotel = journeys["hotelName"].notna()
assert (
    journeys.loc[has_hotel, "destination"].astype(str)
    == journeys.loc[has_hotel, "hotel_place"].astype(str)
).all()

# --------------------------------------------------
# 4. Sort chronologically and split by journey
# --------------------------------------------------
journeys = journeys.sort_values(
    ["departure_date", "travelCode"]
).reset_index(drop=True)

split_idx = int(len(journeys) * 0.8)

# Move the boundary to a date so that journeys departing
# on the same date are not split across train and test.
cutoff = journeys.iloc[split_idx]["departure_date"]

train = journeys[journeys["departure_date"] < cutoff].copy()
test = journeys[journeys["departure_date"] >= cutoff].copy()

assert set(train["travelCode"]).isdisjoint(
    set(test["travelCode"])
)
assert train["departure_date"].max() < test["departure_date"].min()

# --------------------------------------------------
# 5. Save outputs
# --------------------------------------------------
journeys.to_csv(OUT / "journeys.csv", index=False)
train.to_csv(OUT / "journeys_train.csv", index=False)
test.to_csv(OUT / "journeys_test.csv", index=False)

print("=== DATA PREPARATION SUMMARY ===")
print(f"Total journeys: {len(journeys):,}")
print(f"Unique users: {journeys['userCode'].nunique():,}")
print(f"Journeys with hotels: {has_hotel.sum():,}")
print(f"Hotel coverage: {has_hotel.mean():.1%}")
print(f"Training journeys: {len(train):,}")
print(f"Testing journeys: {len(test):,}")
print(f"Chronological cutoff: {cutoff}")
print(
    "Training date range:",
    train["departure_date"].min(),
    "to",
    train["departure_date"].max(),
)
print(
    "Testing date range:",
    test["departure_date"].min(),
    "to",
    test["departure_date"].max(),
)
print(f"Saved datasets to: {OUT.resolve()}")
