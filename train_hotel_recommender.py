#!/usr/bin/env python3
"""
Train the hotel recommender.

The recommender is a per-destination candidate ranker: for a chosen
destination it scores only the hotels historically available AT THAT
destination (hotels never leave their destination), using user profile
(gender, age, company), user trip history, trip context (origin, dates,
flights) and hotel-destination statistics (popularity, price, stay).
Labels are the hotels actually booked. A RandomForest pipeline ranks
candidates at inference time.
"""

from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

BASE_DIR = Path(".")
DATA_DIR = Path("travel_recommender_outputs")
MODEL_DIR = Path("model_outputs")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

USERS_PATH = BASE_DIR / "users.csv"
TRAIN_PATH = DATA_DIR / "journeys_train.csv"
TEST_PATH = DATA_DIR / "journeys_test.csv"

CAT_FEATURES = [
    "gender", "company", "origin", "destination", "hotelName",
]
NUM_FEATURES = [
    "age",
    "user_trip_count",
    "user_avg_total",
    "user_avg_days",
    "user_hotel_stays",
    "user_avg_hotel_total",
    "user_dest_visits",
    "flight_rows",
    "departure_month",
    "departure_dayofweek",
    "has_return",
    "hotel_bookings",
    "hotel_median_price",
    "hotel_median_days",
    "hotel_median_total",
    "hotel_destination_share",
]
FEATURE_COLUMNS = CAT_FEATURES + NUM_FEATURES
LABEL_COLUMN = "label"

MODEL_PATH = MODEL_DIR / "hotel_recommender.joblib"
CATALOG_PATH = MODEL_DIR / "hotel_catalog.csv"
METRICS_PATH = MODEL_DIR / "hotel_recommender_metrics.json"


def load_split(path):
    df = pd.read_csv(path)
    df["userCode"] = df["userCode"].astype(str)
    df["origin"] = df["origin"].astype(str)
    df["destination"] = df["destination"].astype(str)
    df["hotelName"] = df["hotelName"].astype(str)
    df["hotel_place"] = df["hotel_place"].astype(str)
    df["departure_date"] = pd.to_datetime(df["departure_date"], errors="coerce")
    return df


def build_catalog(df):
    """Per-hotel, per-destination stats computed on TRAIN data only."""
    rows = []
    for hotel, sub in df.groupby("hotelName"):
        dest = sub["hotel_place"].mode().iloc[0] if len(sub) else hotel
        bookings = len(sub)
        rows.append({
            "hotelName": hotel,
            "destination": dest,
            "hotel_bookings": bookings,
            "hotel_median_price": float(sub["price"].median()),
            "hotel_median_days": float(sub["days"].median()),
            "hotel_median_total": float(sub["total"].median()),
        })
    catalog = pd.DataFrame(rows)
    dest_totals = catalog.groupby("destination")["hotel_bookings"].sum().rename("dest_bookings")
    catalog = catalog.merge(dest_totals, on="destination", how="left")
    catalog["hotel_destination_share"] = catalog["hotel_bookings"] / catalog["dest_bookings"]
    return catalog.sort_values(["destination", "hotel_bookings"], ascending=[True, False]).reset_index(drop=True)


def user_profiles(df, users):
    df = df.copy()
    df["_is_hotel"] = _has_hotel(df).astype(int)
    df = df.merge(users, left_on="userCode", right_on="code", how="left")
    g = df.groupby("userCode").agg(
        user_trip_count=("destination", "size"),
        user_avg_total=("total", "mean"),
        user_avg_days=("days", "mean"),
        user_hotel_stays=("_is_hotel", "sum"),
        user_dest_visits=("destination", "nunique"),
    )
    g["user_avg_hotel_total"] = df[df["_is_hotel"] == 1].groupby("userCode")["total"].mean()
    g = g.merge(users[["code", "gender", "company", "age"]].rename(columns={"code": "userCode"}),
                on="userCode", how="left")
    g["gender"] = g["gender"].fillna("unknown").astype(str)
    g["company"] = g["company"].fillna("unknown").astype(str)
    g["age"] = pd.to_numeric(g["age"], errors="coerce").fillna(g["age"].median())
    num_cols = ["user_trip_count", "user_avg_total", "user_avg_days",
                "user_hotel_stays", "user_avg_hotel_total", "user_dest_visits", "age"]
    g[num_cols] = g[num_cols].fillna(0.0)
    return g


def feature_rows(bookings_df, profiles, catalog, label_for=None):
    """Build (feature_row, label) pairs: one row per (booking, candidate hotel)."""
    catalog_by_hotel = catalog.set_index("hotelName")
    rows = []
    for _, b in bookings_df.iterrows():
        user_code = str(b["userCode"])
        prof = profiles.loc[user_code] if user_code in profiles.index else None
        month = pd.Timestamp(b["departure_date"]).month if pd.notna(b["departure_date"]) else 6
        dow = pd.Timestamp(b["departure_date"]).dayofweek if pd.notna(b["departure_date"]) else 0
        for hotel in catalog["hotelName"]:
            cat = catalog_by_hotel.loc[hotel]
            row = {
                "__booking_idx": b.name,
                "gender": prof["gender"] if prof is not None else "unknown",
                "company": prof["company"] if prof is not None else "unknown",
                "origin": str(b["origin"]),
                "destination": str(b["destination"]),
                "hotelName": hotel,
                "age": float(prof["age"]) if prof is not None else 0.0,
                "user_trip_count": float(prof["user_trip_count"]) if prof is not None else 0.0,
                "user_avg_total": float(prof["user_avg_total"]) if prof is not None else 0.0,
                "user_avg_days": float(prof["user_avg_days"]) if prof is not None else 0.0,
                "user_hotel_stays": float(prof["user_hotel_stays"]) if prof is not None else 0.0,
                "user_avg_hotel_total": float(prof["user_avg_hotel_total"]) if prof is not None else 0.0,
                "user_dest_visits": float(prof["user_dest_visits"]) if prof is not None else 0.0,
                "flight_rows": float(b["flight_rows"]) if pd.notna(b["flight_rows"]) else 0.0,
                "departure_month": float(month),
                "departure_dayofweek": float(dow),
                "has_return": 1.0 if pd.notna(b["return_date"]) else 0.0,
                "hotel_bookings": float(cat["hotel_bookings"]),
                "hotel_median_price": float(cat["hotel_median_price"]),
                "hotel_median_days": float(cat["hotel_median_days"]),
                "hotel_median_total": float(cat["hotel_median_total"]),
                "hotel_destination_share": float(cat["hotel_destination_share"]),
            }
            if label_for is not None:
                row[LABEL_COLUMN] = int(hotel == label_for(b))
            rows.append(row)
    return pd.DataFrame(rows, columns=["__booking_idx"] + FEATURE_COLUMNS + ([LABEL_COLUMN] if label_for is not None else []))


def evaluate(model, test_bookings, catalog):
    metrics = {}
    pairs = feature_rows(test_bookings, PROFILES, catalog,
                         label_for=lambda b: b["hotelName"])
    X = pairs[FEATURE_COLUMNS]
    proba = model.predict_proba(X)[:, 1]
    pairs = pairs.assign(booking_idx=pairs["__booking_idx"], proba=proba)

    correct_top1, correct_top3, mrr_sum, n = 0, 0, 0.0, 0
    for bi, b in test_bookings.iterrows():
        sub = pairs[pairs["booking_idx"] == bi]
        order = sub.sort_values("proba", ascending=False)
        ranked_hotels = order["hotelName"].tolist()
        true_hotel = b["hotelName"]
        n += 1
        if ranked_hotels and ranked_hotels[0] == true_hotel:
            correct_top1 += 1
        if true_hotel in ranked_hotels[:3]:
            correct_top3 += 1
        if true_hotel in ranked_hotels:
            mrr_sum += 1.0 / (ranked_hotels.index(true_hotel) + 1)

    metrics["unrestricted_top1_accuracy"] = float(correct_top1 / n)
    metrics["unrestricted_hit_at_3"] = float(correct_top3 / n)
    metrics["unrestricted_mrr"] = float(mrr_sum / n)
    metrics["test_bookings"] = int(n)
    metrics["destinations_served"] = int(catalog["destination"].nunique())

    # Destination-constrained inference mode: candidates restricted to destination.
    constrained_ok, constrained_n = 0, 0
    for _, b in test_bookings.iterrows():
        dest = str(b["destination"])
        cands = catalog[catalog["destination"] == dest]["hotelName"].tolist()
        constrained_n += 1
        if cands and b["hotelName"] in cands:
            constrained_ok += 1
    metrics["constrained_coverage"] = float(constrained_ok / constrained_n)
    return metrics


def main() -> int:
    global PROFILES
    train_df = load_split(TRAIN_PATH)
    test_df = load_split(TEST_PATH)
    users = pd.read_csv(USERS_PATH)
    users["code"] = users["code"].astype(str)

def _has_hotel(df):
    return df["hotelName"].fillna("").astype(str).str.strip() != ""


def main() -> int:
    global PROFILES
    train_df = load_split(TRAIN_PATH)
    test_df = load_split(TEST_PATH)
    users = pd.read_csv(USERS_PATH)
    users["code"] = users["code"].astype(str)

    train_bookings = train_df[_has_hotel(train_df)].copy()
    test_bookings = test_df[_has_hotel(test_df)].copy()
    print(f"Train bookings with hotels: {len(train_bookings)}")
    print(f"Test bookings with hotels:  {len(test_bookings)}")

    catalog = build_catalog(train_bookings)
    print(catalog.to_string(index=False))

    global PROFILES
    PROFILES = user_profiles(train_df, users)

    X_train = feature_rows(train_bookings, PROFILES, catalog,
                           label_for=lambda b: b["hotelName"])
    y_train = X_train[LABEL_COLUMN].to_numpy()
    X_train = X_train[FEATURE_COLUMNS]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
            ("num", SimpleImputer(strategy="constant", fill_value=0.0), NUM_FEATURES),
        ],
        remainder="drop",
    )
    model = Pipeline([
        ("preprocess", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=150, max_depth=None, max_features="sqrt",
            class_weight="balanced", n_jobs=-1, random_state=42)),
    ])

    print("Fitting hotel recommender ...")
    model.fit(X_train, y_train)

    metrics = evaluate(model, test_bookings, catalog)
    print("\n=== EVALUATION ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    joblib.dump(model, MODEL_PATH)
    catalog.to_csv(CATALOG_PATH, index=False)
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved catalog -> {CATALOG_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
