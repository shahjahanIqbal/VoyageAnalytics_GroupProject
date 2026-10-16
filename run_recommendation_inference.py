#!/usr/bin/env python3
"""
End-to-end inference pipeline for VoyageAnalytics travel recommendation.

Usage:
    python run_recommendation_inference.py --user_code 0 --origin "Recife (PE)" --departure_date "2024-07-15"
    python run_recommendation_inference.py --user_code 0 --origin "Recife (PE)" --departure_date "2024-07-15" --destination "Salvador (BH)"
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = Path("model_outputs")
DATA_DIR = Path("travel_recommender_outputs")


def resolve_history_path():
    """Prefer the hotel-enriched journeys file, fall back to journeys.csv."""
    preferred = DATA_DIR / "journeys_with_hotels.csv"
    if preferred.exists():
        return preferred
    return DATA_DIR / "journeys.csv"

# --------------------------------------------------
# Load models
# --------------------------------------------------
def load_destination_model():
    """Load the best destination recommendation model."""
    model_data = joblib.load(MODEL_DIR / "best_destination_model.joblib")
    return model_data

def load_hotel_model():
    """Load the best hotel recommendation model."""
    model_data = joblib.load(MODEL_DIR / "best_hotel_model.joblib")
    return model_data

# --------------------------------------------------
# Build features for destination recommendation
# --------------------------------------------------
def build_destination_features(user_code, origin, departure_date, candidates, history_df, seasonal_features):
    """Build candidate-level features for a single user query."""
    departure_date = pd.to_datetime(departure_date)
    month = departure_date.month
    weekday = departure_date.dayofweek
    
    # Ensure history_df has string userCode
    history_df = history_df.copy()
    history_df["userCode"] = history_df["userCode"].astype(str)
    history_df["origin"] = history_df["origin"].astype(str)
    history_df["destination"] = history_df["destination"].astype(str)
    
    # Historical counts from training data
    user_destination = history_df.groupby(["userCode", "destination"]).size().to_dict()
    user_route = history_df.groupby(["userCode", "origin", "destination"]).size().to_dict()
    origin_destination = history_df.groupby(["origin", "destination"]).size().to_dict()
    destination_counts = history_df["destination"].value_counts().to_dict()
    user_trip_counts = history_df["userCode"].value_counts().to_dict()
    user_origin_counts = history_df.groupby(["userCode", "origin"]).size().to_dict()
    origin_counts = history_df["origin"].value_counts().to_dict()
    
    user_code_str = str(user_code)
    origin_str = str(origin)
    
    records = []
    for candidate in candidates:
        records.append({
            "userCode": user_code_str,
            "origin": origin_str,
            "candidate": str(candidate),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "day_of_week": weekday,
            "month": month,
            "user_candidate_count": user_destination.get((str(user_code), candidate), 0),
            "user_route_count": user_route.get((str(user_code), origin, candidate), 0),
            "origin_candidate_count": origin_destination.get((origin, candidate), 0),
            "candidate_popularity": destination_counts.get(candidate, 0),
            "user_history_count": user_trip_counts.get(str(user_code), 0),
            "user_origin_history_count": user_origin_counts.get((str(user_code), origin), 0),
            "origin_history_count": origin_counts.get(origin, 0),
        })
    
    df = pd.DataFrame(records)
    
    # Add seasonal features
    df = add_seasonal_features_inference(df, seasonal_features)
    
    return df

def add_seasonal_features_inference(df, seasonal_features):
    """Add seasonal features to inference dataframe."""
    df = df.copy()
    
    # Merge seasonal counts
    seasonal = seasonal_features["seasonal_counts"].rename(columns={
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
    user_month = seasonal_features["user_month_totals"].rename(columns={"userCode": "userCode", "month": "month"})
    df = df.merge(user_month, on=["userCode", "month"], how="left")
    df["user_month_total"] = df["user_month_total"].fillna(0).astype(int)
    
    # User destination totals
    user_dest = seasonal_features["user_dest_totals"].rename(columns={"userCode": "userCode", "destination": "candidate"})
    df = df.merge(user_dest, on=["userCode", "candidate"], how="left")
    df["user_dest_total"] = df["user_dest_total"].fillna(0).astype(int)
    
    # Global month-destination rate
    global_md = seasonal_features["global_month_dest"][["month", "candidate", "global_month_dest_rate"]]
    df = df.merge(global_md, on=["month", "candidate"], how="left")
    df["global_month_dest_rate"] = df["global_month_dest_rate"].fillna(0)
    
    # Engineered seasonal features
    smoothing = 5.0
    df["seasonal_strength"] = (df["seasonal_trips"] + smoothing) / (df["user_month_total"] + smoothing)
    df["is_recurring_seasonal"] = ((df["seasonal_trips"] >= 2) & (df["seasonal_years"] >= 2)).astype(int)
    df["user_dest_month_share"] = df["seasonal_trips"] / (df["user_dest_total"] + 1)
    df["dest_month_popularity"] = df["global_month_dest_rate"]
    
    return df

def load_seasonal_features(split="user_80_20"):
    """Load seasonal features from training data."""
    if split == "user_80_20":
        train_df = pd.read_csv(DATA_DIR / "user_train.csv")
    else:
        train_df = pd.read_csv(DATA_DIR / "global_train.csv")
    
    train_df["departure_date"] = pd.to_datetime(train_df["departure_date"])
    train_df["userCode"] = train_df["userCode"].astype(str)
    train_df["destination"] = train_df["destination"].astype(str)
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

# --------------------------------------------------
# Destination recommendation
# --------------------------------------------------
def recommend_destinations(user_code, origin, departure_date, top_k=5, split="user_80_20"):
    """Recommend top-K destinations for a user."""
    model_data = load_destination_model()
    model = model_data["model"]
    features = model_data["features"]
    candidates = model_data["candidates"]
    
    # Load history data
    if split == "user_80_20":
        history_df = pd.read_csv(DATA_DIR / "user_train.csv")
    else:
        history_df = pd.read_csv(DATA_DIR / "global_train.csv")
    
    history_df["userCode"] = history_df["userCode"].astype(str)
    history_df["origin"] = history_df["origin"].astype(str)
    history_df["destination"] = history_df["destination"].astype(str)
    
    # Load seasonal features
    seasonal_features = load_seasonal_features(split)
    
    # Build features
    X = build_destination_features(user_code, origin, departure_date, candidates, history_df, seasonal_features)
    
    # Predict
    scores = model.predict_proba(X[features])[:, 1]
    
    # Rank
    results = pd.DataFrame({
        "destination": candidates,
        "score": scores
    }).sort_values("score", ascending=False).head(top_k)
    
    return results.reset_index(drop=True)

# --------------------------------------------------
# Hotel recommendation
# --------------------------------------------------
def build_hotel_features(user_code, destination, departure_date, history_df):
    """Build features for hotel expenditure prediction."""
    departure_date = pd.to_datetime(departure_date)
    month = departure_date.month
    
    # User historical features
    user_past = history_df[history_df["userCode"] == str(user_code)]
    
    # User-destination historical features
    user_dest_past = user_past[user_past["destination"] == destination]
    
    # Hotel destination features
    hotel_dest = history_df[history_df["destination"] == destination]
    
    # Most common hotel for this user at this destination
    if len(user_dest_past) > 0:
        most_common_hotel = user_dest_past["hotelName"].mode().iloc[0]
        user_dest_avg_hotel_price = user_dest_past["price"].mean()
        user_dest_avg_hotel_total = user_dest_past["total"].mean()
        user_dest_past_hotel_stays = len(user_dest_past)
    else:
        most_common_hotel = "none"
        user_dest_avg_hotel_price = 0
        user_dest_avg_hotel_total = 0
        user_dest_past_hotel_stays = 0
    
    # User overall features
    user_past_trip_count = len(user_past)
    user_past_total_expenditure = user_past["total"].sum()
    user_past_avg_expenditure = user_past["total"].mean() if len(user_past) > 0 else 0
    user_past_hotel_expenditure = user_past["total"].sum()
    user_past_avg_hotel_expenditure = user_past["total"].mean() if len(user_past) > 0 else 0
    user_past_flight_expenditure = 0
    user_past_avg_flight_expenditure = 0
    user_preferred_flight_class = "unknown"
    user_preferred_agency = "unknown"
    user_unique_destinations = user_past["destination"].nunique()
    user_unique_origins = user_past["origin"].nunique()
    
    # Destination hotel features
    dest_avg_hotel_price = hotel_dest["price"].mean() if len(hotel_dest) > 0 else 0
    dest_avg_hotel_total = hotel_dest["total"].mean() if len(hotel_dest) > 0 else 0
    dest_avg_hotel_days = hotel_dest["days"].mean() if len(hotel_dest) > 0 else 0
    dest_hotel_count = hotel_dest["hotelName"].nunique() if len(hotel_dest) > 0 else 0
    
    return {
        "user_past_trip_count": user_past_trip_count,
        "user_past_total_expenditure": user_past_total_expenditure,
        "user_past_avg_expenditure": user_past_avg_expenditure,
        "user_past_hotel_expenditure": user_past_hotel_expenditure,
        "user_past_avg_hotel_expenditure": user_past_avg_hotel_expenditure,
        "user_past_flight_expenditure": user_past_flight_expenditure,
        "user_past_avg_flight_expenditure": user_past_avg_flight_expenditure,
        "user_preferred_flight_class": str(user_preferred_flight_class),
        "user_preferred_agency": str(user_preferred_agency),
        "user_unique_destinations": user_unique_destinations,
        "user_unique_origins": user_unique_origins,
        "user_dest_past_hotel_stays": user_dest_past_hotel_stays,
        "user_dest_most_common_hotel": str(most_common_hotel),
        "user_dest_avg_hotel_price": user_dest_avg_hotel_price,
        "user_dest_avg_hotel_total": user_dest_avg_hotel_total,
        "dest_avg_hotel_price": dest_avg_hotel_price,
        "dest_avg_hotel_total": dest_avg_hotel_total,
        "dest_avg_hotel_days": dest_avg_hotel_days,
        "dest_hotel_count": dest_hotel_count,
        "departure_month": month,
        "departure_month_sin": np.sin(2 * np.pi * month / 12),
        "departure_month_cos": np.cos(2 * np.pi * month / 12),
        "departure_day_of_week": departure_date.dayofweek,
    }

def predict_hotel_expenditure(user_code, destination, departure_date):
    """Predict hotel expenditure for a user-destination pair."""
    model_data = load_hotel_model()
    model = model_data["model"]
    features = model_data["features"]
    
    # Load full history data
    history_df = pd.read_csv(resolve_history_path())
    history_df["departure_date"] = pd.to_datetime(history_df["departure_date"])
    history_df["userCode"] = history_df["userCode"].astype(str)
    history_df["destination"] = history_df["destination"].astype(str)
    history_df["origin"] = history_df["origin"].astype(str)
    
    # Build features
    feat_dict = build_hotel_features(user_code, destination, departure_date, history_df)
    X = pd.DataFrame([feat_dict])
    
    # Ensure all features present
    for f in features:
        if f not in X.columns:
            X[f] = 0
    
    X = X[features]
    
    # Predict
    prediction = model.predict(X)[0]
    
    return prediction

# --------------------------------------------------
# Main inference function
# --------------------------------------------------
def run_inference(user_code, origin, departure_date, destination=None, top_k=5):
    """Run the full recommendation pipeline."""
    print(f"\n{'='*60}")
    print(f"VOYAGE ANALYTICS - TRAVEL RECOMMENDATION")
    print(f"{'='*60}")
    print(f"User: {user_code}")
    print(f"Origin: {origin}")
    print(f"Departure Date: {departure_date}")
    print(f"{'='*60}")
    
    # Step 1: Destination recommendation
    print(f"\n[1/2] Recommending destinations...")
    dest_results = recommend_destinations(user_code, origin, departure_date, top_k=top_k)
    
    print(f"\nTop {top_k} Recommended Destinations:")
    for i, row in dest_results.iterrows():
        print(f"  {i+1}. {row['destination']} (score: {row['score']:.4f})")
    
    # Step 2: Hotel recommendation
    if destination is None:
        destination = dest_results.iloc[0]["destination"]
        print(f"\n[2/2] Predicting hotel expenditure for top destination: {destination}")
    else:
        print(f"\n[2/2] Predicting hotel expenditure for selected destination: {destination}")
    
    predicted_hotel_total = predict_hotel_expenditure(user_code, destination, departure_date)
    
    print(f"\nPredicted Hotel Expenditure: ${predicted_hotel_total:.2f}")
    
    # Also show the actual hotel for this destination from training data
    history_df = pd.read_csv(resolve_history_path())
    history_df["departure_date"] = pd.to_datetime(history_df["departure_date"])
    history_df["userCode"] = history_df["userCode"].astype(str)
    history_df["destination"] = history_df["destination"].astype(str)
    
    user_dest_hotels = history_df[
        (history_df["userCode"] == str(user_code)) & 
        (history_df["destination"] == destination)
    ]["hotelName"].value_counts()
    
    if len(user_dest_hotels) > 0:
        print(f"\nUser's Historical Hotels at {destination}:")
        for hotel, count in user_dest_hotels.items():
            print(f"  - {hotel}: {count} stays")
    
    # Most common hotel overall at this destination
    dest_hotels = history_df[history_df["destination"] == destination]["hotelName"].value_counts()
    if len(dest_hotels) > 0:
        print(f"\nMost Popular Hotels at {destination}:")
        for hotel, count in dest_hotels.head(3).items():
            print(f"  - {hotel}: {count} bookings")
    
    print(f"\n{'='*60}")
    
    return {
        "destination_recommendations": dest_results.to_dict("records"),
        "selected_destination": destination,
        "predicted_hotel_total": predicted_hotel_total,
    }

# --------------------------------------------------
# CLI
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="VoyageAnalytics Travel Recommendation Inference")
    parser.add_argument("--user_code", type=str, required=True, help="User code")
    parser.add_argument("--origin", type=str, required=True, help="Origin city")
    parser.add_argument("--departure_date", type=str, required=True, help="Departure date (YYYY-MM-DD)")
    parser.add_argument("--destination", type=str, help="Specific destination (optional, uses top-1 if not provided)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of destination recommendations")
    
    args = parser.parse_args()
    
    result = run_inference(
        user_code=args.user_code,
        origin=args.origin,
        departure_date=args.departure_date,
        destination=args.destination,
        top_k=args.top_k,
    )
    
    # Save result
    import json
    output_path = MODEL_DIR / f"inference_result_{args.user_code}_{args.departure_date}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResult saved to: {output_path}")

if __name__ == "__main__":
    main()