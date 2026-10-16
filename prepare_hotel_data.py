import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(".")
OUT = Path("model_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Load and validate datasets
# --------------------------------------------------
journeys = pd.read_csv(DATA_DIR / "travel_recommender_outputs" / "journeys.csv")
hotels = pd.read_csv(DATA_DIR / "hotels.csv")
users = pd.read_csv(DATA_DIR / "users.csv")

for df in (journeys, hotels, users):
    df.columns = df.columns.str.strip()

journeys["departure_date"] = pd.to_datetime(journeys["departure_date"], errors="coerce")
hotels["date"] = pd.to_datetime(hotels["date"], errors="coerce")

# Standardize column names
hotels = hotels.rename(columns={"name": "hotelName"})
users = users.rename(columns={"code": "userCode", "name": "user_name"})

# Keep only journeys with hotel bookings
journeys_with_hotel = journeys[journeys["hotelName"].notna()].copy()
print(f"Total journeys: {len(journeys)}")
print(f"Journeys with hotel: {len(journeys_with_hotel)}")

# --------------------------------------------------
# 2. Create user-level features from journey history
# --------------------------------------------------
# User's travel history before each journey
journeys_with_hotel = journeys_with_hotel.sort_values(["userCode", "departure_date"]).reset_index(drop=True)

user_features_list = []
for _, row in journeys_with_hotel.iterrows():
    user = row["userCode"]
    dep_date = row["departure_date"]
    
    # Historical journeys for this user before this date
    past = journeys_with_hotel[
        (journeys_with_hotel["userCode"] == user) & 
        (journeys_with_hotel["departure_date"] < dep_date)
    ]
    
    user_features_list.append({
        "travelCode": row["travelCode"],
        "userCode": user,
        "user_past_trip_count": len(past),
        "user_past_total_expenditure": past["total"].sum() if "total" in past.columns else 0,
        "user_past_avg_expenditure": past["total"].mean() if "total" in past.columns and len(past) > 0 else 0,
        "user_past_hotel_expenditure": past["hotel_total"].sum() if "hotel_total" in past.columns else 0,
        "user_past_avg_hotel_expenditure": past["hotel_total"].mean() if "hotel_total" in past.columns and len(past) > 0 else 0,
        "user_past_flight_expenditure": past["price"].sum() if "price" in past.columns else 0,
        "user_past_avg_flight_expenditure": past["price"].mean() if "price" in past.columns and len(past) > 0 else 0,
        "user_preferred_flight_class": past["flight_class"].mode().iloc[0] if "flight_class" in past.columns and len(past) > 0 else "unknown",
        "user_preferred_agency": past["agency"].mode().iloc[0] if "agency" in past.columns and len(past) > 0 else "unknown",
        "user_unique_destinations": past["destination"].nunique(),
        "user_unique_origins": past["origin"].nunique(),
    })

user_features = pd.DataFrame(user_features_list)

# --------------------------------------------------
# 3. Create destination-level hotel features from hotel data
# --------------------------------------------------
# Hotel data per destination
hotel_dest_features = hotels.groupby("place").agg(
    dest_avg_hotel_price=("price", "mean"),
    dest_avg_hotel_total=("total", "mean"),
    dest_avg_hotel_days=("days", "mean"),
    dest_hotel_count=("hotelName", "nunique"),
).reset_index()
hotel_dest_features = hotel_dest_features.rename(columns={"place": "destination"})

# --------------------------------------------------
# 4. Create user-destination historical hotel features
# --------------------------------------------------
user_dest_hotel_history = []
for _, row in journeys_with_hotel.iterrows():
    user = row["userCode"]
    destination = row["destination"]
    dep_date = row["departure_date"]
    
    # Past hotel stays for this user at this destination
    past_hotels = hotels[
        (hotels["userCode"] == user) & 
        (hotels["place"] == destination) & 
        (hotels["date"] < dep_date)
    ]
    
    if len(past_hotels) > 0:
        most_common_hotel = past_hotels["hotelName"].mode().iloc[0]
        avg_hotel_price = past_hotels["price"].mean()
        avg_hotel_total = past_hotels["total"].mean()
        hotel_stay_count = len(past_hotels)
    else:
        most_common_hotel = "none"
        avg_hotel_price = 0
        avg_hotel_total = 0
        hotel_stay_count = 0
    
    user_dest_hotel_history.append({
        "travelCode": row["travelCode"],
        "userCode": user,
        "destination": destination,
        "user_dest_past_hotel_stays": hotel_stay_count,
        "user_dest_most_common_hotel": most_common_hotel,
        "user_dest_avg_hotel_price": avg_hotel_price,
        "user_dest_avg_hotel_total": avg_hotel_total,
    })

user_dest_hotel_df = pd.DataFrame(user_dest_hotel_history)
if "userCode" not in user_dest_hotel_df.columns:
    user_dest_hotel_df["userCode"] = user_dest_hotel_df["travelCode"].map(
        journeys_with_hotel.set_index("travelCode")["userCode"]
    )

# --------------------------------------------------
# 5. Build final training dataset
# --------------------------------------------------
# Start with journeys that have hotels
df = journeys_with_hotel.merge(user_features, on="travelCode", how="left")
df = df.merge(user_dest_hotel_df, on=["travelCode", "destination"], how="left")
df = df.merge(hotel_dest_features, on="destination", how="left")
df = df.merge(users[["userCode", "user_name", "gender", "age", "company"]], on="userCode", how="left")

# Hotel target variables already in journeys_with_hotel from original join
# Rename for clarity
df = df.rename(columns={
    "hotelName": "hotelName",
    "price": "hotel_price",
    "total": "hotel_total",
    "days": "hotel_days",
})

# Add calendar features
df["departure_month"] = df["departure_date"].dt.month
df["departure_month_sin"] = np.sin(2 * np.pi * df["departure_month"] / 12)
df["departure_month_cos"] = np.cos(2 * np.pi * df["departure_month"] / 12)
df["departure_day_of_week"] = df["departure_date"].dt.dayofweek

# Add flight features
if "flight_class" in df.columns:
    df["flight_class"] = df["flight_class"].astype(str)
if "agency" in df.columns:
    df["agency"] = df["agency"].astype(str)
if "distance" in df.columns:
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
if "price" in df.columns:
    df["flight_price"] = pd.to_numeric(df["price"], errors="coerce")

# Fill missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

# --------------------------------------------------
# 6. User-chronological split (80/20)
# --------------------------------------------------
df = df.sort_values(["userCode", "departure_date"]).reset_index(drop=True)

train_parts = []
test_parts = []

for _, group in df.groupby("userCode", sort=False):
    group = group.sort_values(["departure_date", "travelCode"])
    n = len(group)
    
    if n < 2:
        train_parts.append(group)
        continue
    
    n_train = max(1, int(n * 0.8))
    n_train = min(n_train, n - 1)
    
    train_parts.append(group.iloc[:n_train])
    test_parts.append(group.iloc[n_train:])

train_df = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0].copy()
test_df = pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0].copy()

train_df = train_df.sort_values(["departure_date", "travelCode"]).reset_index(drop=True)
test_df = test_df.sort_values(["departure_date", "travelCode"]).reset_index(drop=True)

print(f"\n=== SPLIT SUMMARY ===")
print(f"Training samples: {len(train_df)}")
print(f"Testing samples:  {len(test_df)}")
print(f"Training users:   {train_df['userCode'].nunique()}")
print(f"Testing users:    {test_df['userCode'].nunique()}")
print(f"Training destinations: {train_df['destination'].nunique()}")
print(f"Testing destinations:  {test_df['destination'].nunique()}")
print(f"Unique hotels in train: {train_df['hotelName'].nunique()}")
print(f"Unique hotels in test:  {test_df['hotelName'].nunique()}")

# Check target distribution
print(f"\nHotel total (train): mean={train_df['hotel_total'].mean():.2f}, std={train_df['hotel_total'].std():.2f}")
print(f"Hotel total (test):  mean={test_df['hotel_total'].mean():.2f}, std={test_df['hotel_total'].std():.2f}")

# --------------------------------------------------
# 7. Save outputs
# --------------------------------------------------
train_df.to_csv(OUT / "hotel_training_data.csv", index=False)
train_df.to_csv(OUT / "hotel_train.csv", index=False)
test_df.to_csv(OUT / "hotel_test.csv", index=False)

# Save feature list for reference
feature_cols = [c for c in train_df.columns if c not in [
    "travelCode", "userCode", "origin", "destination", "departure_date",
    "hotelName", "hotel_price", "hotel_days", "hotel_total",
    "return_origin", "return_destination", "return_date",
    "flight_rows", "hotel_place", "hotel_date", "hotel_userCode",
    "user_name", "gender", "age", "company"
]]
pd.Series(feature_cols).to_csv(OUT / "hotel_feature_columns.csv", index=False, header=["feature"])

print(f"\nSaved to: {OUT.resolve()}")
print(f"  hotel_training_data.csv ({len(train_df)} rows)")
print(f"  hotel_train.csv ({len(train_df)} rows)")
print(f"  hotel_test.csv ({len(test_df)} rows)")
print(f"  hotel_feature_columns.csv ({len(feature_cols)} features)")