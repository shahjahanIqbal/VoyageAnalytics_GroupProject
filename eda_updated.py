#!/usr/bin/env python
"""
Voyage Analytics - merged EDA + ML training + leakage analysis
================================================================

Merges the two previously separate efforts into one reproducible script:

  * EDA  -> ../voyage/eda.py            (users / hotels / flights analysis)
  * ML   -> ../../VoyageAnalytics_GroupProject/flight price prediction.ipynb
            (label encoding, train/test split, LinearRegression /
             DecisionTree / RandomForest, joblib export)

Design goals
------------
1. One shared preprocessing layer. ``clean_dataframe`` and
   ``add_date_features`` are defined once and consumed by *both* the
   EDA section and the ML section. No section re-implements loading,
   cleaning, date parsing, monthly roll-ups, metric computation or
   figure saving.
2. Dedicated sections:
     Section 1  Flight data analysis        (EDA on flights.csv)
     Section 2  Hotel & user analysis       (supporting EDA)
     Section 3  Leakage analysis            (target/config cardinality)
     Section 4  ML training & evaluation    (training + honest metrics)
3. Non-destructive outputs. Everything this script writes goes to
   ``updated_eda/eda_outputs`` and ``updated_eda/ml_artifacts``.
   The existing ``../voyage/eda_outputs`` is never read or written.

Dataset caveats that this script measures rather than assumes
------------------------------------------------------------
* ``flightType`` is flight class (economic / premium / firstClass).
  It is NOT trip purpose. There is no business/pleasure field.
* All observed cities are domestic Brazilian cities, so no
  international/domestic flag can be derived.
* Adaptive pricing is treated as exploratory only: the data has no
  inventory, lead-time, competitor price or cancellation columns.

Usage
-----
    python merged_eda.py                 # full run
    python merged_eda.py --skip-geo      # no basemap download
    python merged_eda.py --max-rows 50000  # fast smoke run
    python merged_eda.py --no-save-model   # evaluate only
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless: required for servers / CI
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    GroupKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import (  # noqa: E402
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.tree import DecisionTreeRegressor  # noqa: E402


# =====================================================================
# SECTION 0 - Configuration
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent                      # ../voyage
OUTPUT_DIR = BASE_DIR / "eda_outputs"           # NEW output directory
ARTIFACT_DIR = BASE_DIR / "ml_artifacts"        # NEW model directory
GEO_CACHE_DIR = DATA_DIR / "geo_cache"          # reuse existing download

USERS_FILE = DATA_DIR / "users.csv"
HOTELS_FILE = DATA_DIR / "hotels.csv"
FLIGHTS_FILE = DATA_DIR / "flights.csv"

SEED = 42
TEST_SIZE = 0.30                 # matches the original notebook
TARGET = "price"

# Identifiers and raw date string are never model features.
NON_FEATURE_COLUMNS = ["travelCode", "userCode", "date"]

# The exact feature set the saved RandomForest was trained on.
MODEL_FEATURES = [
    "from", "to", "flightType", "time", "distance",
    "agency", "year", "month", "day",
]
CATEGORICAL_FEATURES = ["from", "to", "flightType", "agency"]
NUMERIC_FEATURES = ["time", "distance", "year", "month", "day"]

# --- Leakage analysis column groups (as specified) --------------------
GROUP_COLS_WITH_DATE = [
    "from", "to", "flightType", "time", "distance",
    "agency", "year", "month", "day",
]
GROUP_COLS_NO_DATE = [
    "from", "to", "flightType", "time", "distance", "agency",
]

# Coordinates for the nine cities present in the dataset. Used only for
# the geographic-frequency visualisation.
CITY_COORDS = {
    "Recife (PE)": (-8.0476, -34.8770),
    "Florianopolis (SC)": (-27.5949, -48.5482),
    "Brasilia (DF)": (-15.7939, -47.8828),
    "Aracaju (SE)": (-10.9472, -37.0731),
    "Salvador (BH)": (-12.9777, -38.5016),
    "Campo Grande (MS)": (-20.4697, -54.6201),
    "Sao Paulo (SP)": (-23.5505, -46.6333),
    "Natal (RN)": (-5.7945, -35.2110),
    "Rio de Janeiro (RJ)": (-22.9068, -43.1729),
}

# Years with a complete calendar in the supplied data, for seasonality.
COMPLETE_YEARS = [2020, 2021, 2022]


# =====================================================================
# SECTION 0b - Shared primitives (used by every section below)
# =====================================================================

def banner(title: str, char: str = "=") -> None:
    """Print a section banner so long runs stay navigable."""
    line = char * 78
    print(f"\n{line}\n{title}\n{line}")


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, name: str) -> Path:
    """Persist a DataFrame to the new output directory."""
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False)
    print(f"    saved {name}  ({len(df):,} rows)")
    return path


def save_fig(fig, name: str, dpi: int = 150) -> Path:
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {name}")
    return path


def thinned_ticks(n: int, every: int = 3) -> np.ndarray:
    """Tick positions for long category axes."""
    return np.arange(0, n, every)


def month_key(series: pd.Series) -> pd.Series:
    """'YYYY-MM' string key from a datetime column."""
    return series.dt.to_period("M").astype(str)


def regression_metrics(y_true, y_pred) -> dict:
    """Single metric implementation reused by every model evaluation."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred

    safe = np.where(y_true == 0, np.nan, y_true)
    return {
        "MAE": float(np.mean(np.abs(resid))),
        "RMSE": float(np.sqrt(np.mean(resid ** 2))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_pct": float(np.nanmean(np.abs(resid / safe)) * 100),
        "MaxAbsError": float(np.max(np.abs(resid))),
        "n_test": int(y_true.size),
    }


# =====================================================================
# SECTION 0c - Loading, cleaning, feature engineering (shared)
# =====================================================================

def load_data(max_rows: int | None = None) -> dict:
    """Read the three source files. Single definition, reused everywhere."""
    frames = {
        "users": pd.read_csv(USERS_FILE),
        "hotels": pd.read_csv(HOTELS_FILE),
        "flights": pd.read_csv(FLIGHTS_FILE),
    }
    if max_rows is not None:
        frames = {k: v.head(max_rows).copy() for k, v in frames.items()}
    return frames


def clean_dataframe(df: pd.DataFrame, date_columns=None) -> pd.DataFrame:
    """
    Generic cleaning, defined once and shared by EDA and ML:
      - strip column names
      - strip whitespace from string values
      - parse requested date columns
      - remove exact duplicate rows
      - median imputation for numeric columns
      - mode imputation for categorical columns
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    df = df.drop_duplicates().reset_index(drop=True)

    for col in date_columns or []:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        if df[col].isna().any():
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "Unknown")

    return df


def add_date_features(
    df: pd.DataFrame,
    date_col: str = "date",
    include: tuple = ("year", "month", "month_name", "quarter",
                      "day_of_week", "is_weekend"),
) -> pd.DataFrame:
    """
    Calendar features, defined once.

    ``include`` lets the ML path request only the numeric parts it needs
    (``year``, ``month``, ``day``) without inheriting EDA-only columns,
    which avoids a second, near-identical date-parsing implementation.
    """
    df = df.copy()
    dt = df[date_col].dt

    builders = {
        "year": lambda: dt.year,
        "month": lambda: dt.month,
        "month_name": lambda: dt.month_name().str[:3],
        "quarter": lambda: dt.quarter,
        "day_of_week": lambda: dt.day_name(),
        "is_weekend": lambda: dt.dayofweek >= 5,
        "day": lambda: dt.day,
        "dayofweek": lambda: dt.dayofweek,
    }
    for name in include:
        df[name] = builders[name]()
    return df


def build_model_frame(flights: pd.DataFrame) -> pd.DataFrame:
    """
    Exact feature matrix the saved RandomForest was trained on.

    Reused by: the leakage analysis, the notebook reproduction, the honest
    evaluation and the demonstration prediction.
    """
    frame = add_date_features(flights, include=("year", "month", "day"))
    drop = NON_FEATURE_COLUMNS + [c for c in frame.columns
                                 if c not in MODEL_FEATURES + [TARGET]]
    frame = frame.drop(columns=drop, errors="ignore")
    return frame[MODEL_FEATURES + [TARGET]]


# =====================================================================
# SECTION 1 - Flight data analysis
# =====================================================================

def flight_data_quality(flights: pd.DataFrame, hotels: pd.DataFrame) -> pd.DataFrame:
    """Column-level quality profile for the flight and hotel tables."""
    rows = []
    for name, df in {"flights": flights, "hotels": hotels}.items():
        for col in df.columns:
            rows.append({
                "dataset": name,
                "column": col,
                "dtype": str(df[col].dtype),
                "rows": len(df),
                "missing": int(df[col].isna().sum()),
                "missing_pct": round(float(df[col].isna().mean()) * 100, 3),
                "unique": int(df[col].nunique(dropna=True)),
            })
    report = pd.DataFrame(rows)
    save_table(report, "data_quality.csv")
    return report


def plot_price_distribution(flights: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(flights[TARGET], bins=30)
    ax.set_title("Flight Price Distribution")
    ax.set_xlabel("Flight price")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    save_fig(fig, "flight_price_distribution.png")


def plot_destination_frequency(flights: pd.DataFrame) -> pd.DataFrame:
    dest = flights["to"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    dest.plot(kind="barh", ax=ax)
    ax.set_title("Flight Destination Frequency")
    ax.set_xlabel("Number of flight records")
    ax.set_ylabel("Destination")
    fig.tight_layout()
    save_fig(fig, "flight_destination_frequency.png")
    return dest.rename_axis("destination").reset_index(name="flights")


def plot_price_by_class_and_agency(flights: pd.DataFrame) -> None:
    """Flight class is NOT trip purpose - see module docstring."""
    for column, title, fname in [
        ("flightType", "Flight Price Variation by Flight Class",
         "flight_price_by_class.png"),
        ("agency", "Flight Price Variation by Agency",
         "flight_price_by_agency.png"),
    ]:
        levels = flights[column].unique()
        data = [flights.loc[flights[column] == lvl, TARGET] for lvl in levels]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(data, tick_labels=list(levels))
        ax.set_title(title)
        ax.set_xlabel(column)
        ax.set_ylabel("Price")
        fig.tight_layout()
        save_fig(fig, fname)


def plot_distance_price(flights: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(flights["distance"], flights[TARGET], alpha=0.15, s=8)
    ax.set_title("Flight Distance vs Price")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Price")
    fig.tight_layout()
    save_fig(fig, "distance_vs_price.png")


def plot_flight_correlation(flights: pd.DataFrame) -> None:
    corr = flights[[TARGET, "time", "distance"]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(corr.values, aspect="auto", cmap="coolwarm",
                      vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.columns)
    ax.set_title("Flight Numerical Feature Correlations")
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Pearson correlation")
    fig.tight_layout()
    save_fig(fig, "correlation_flights.png")


def plot_monthly_travel_demand(flights: pd.DataFrame,
                               hotels: pd.DataFrame) -> pd.DataFrame:
    flight_monthly = (flights.assign(month=month_key(flights["date"]))
                      .groupby("month")["travelCode"].nunique()
                      .rename("flight_trips"))
    hotel_monthly = (hotels.assign(month=month_key(hotels["date"]))
                     .groupby("month").size().rename("hotel_bookings"))

    monthly = pd.concat([flight_monthly, hotel_monthly], axis=1).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly.index, monthly["flight_trips"], marker="o",
            label="Flight trips")
    ax.plot(monthly.index, monthly["hotel_bookings"], marker="o",
            label="Hotel bookings")
    ax.set_title("Monthly Travel Demand")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of trips / bookings")
    ticks = thinned_ticks(len(monthly.index))
    ax.set_xticks(ticks)
    ax.set_xticklabels(monthly.index[ticks], rotation=45, ha="right")
    ax.legend(title="Measure", frameon=True)
    fig.tight_layout()
    save_fig(fig, "monthly_travel_demand.png")
    return monthly


def plot_monthly_expenditure(flights: pd.DataFrame,
                             hotels: pd.DataFrame) -> pd.DataFrame:
    monthly = pd.concat([
        flights.assign(month=month_key(flights["date"]))
        .groupby("month")[TARGET].sum().rename("flight_expenditure"),
        hotels.assign(month=month_key(hotels["date"]))
        .groupby("month")["total"].sum().rename("hotel_expenditure"),
    ], axis=1).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly.index, monthly["flight_expenditure"], marker="o",
            label="Flight expenditure")
    ax.plot(monthly.index, monthly["hotel_expenditure"], marker="o",
            label="Hotel expenditure")
    ax.set_title("Monthly Travel Expenditure")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total expenditure")
    ticks = thinned_ticks(len(monthly.index))
    ax.set_xticks(ticks)
    ax.set_xticklabels(monthly.index[ticks], rotation=45, ha="right")
    ax.legend(title="Measure", frameon=True)
    fig.tight_layout()
    save_fig(fig, "monthly_expenditure.png")
    return monthly


def plot_price_demand_relationship(flights: pd.DataFrame) -> pd.DataFrame:
    monthly = (flights.assign(month=month_key(flights["date"]))
               .groupby("month")
               .agg(demand=("travelCode", "nunique"),
                    total_records=("travelCode", "size"),
                    average_price=(TARGET, "mean"))
               .reset_index())

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(monthly["month"], monthly["demand"], marker="o",
             label="Travel demand")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Number of trips")
    ticks = thinned_ticks(len(monthly.index))
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(monthly.index[ticks], rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(monthly["month"], monthly["average_price"], marker="s",
             label="Average flight price")
    ax2.set_ylabel("Average flight price")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)
    ax1.set_title("Flight Demand and Average Price Over Time")
    fig.tight_layout()
    save_fig(fig, "monthly_price_demand.png")

    save_table(monthly, "monthly_price_demand.csv")
    return monthly


def plot_seasonality(flights: pd.DataFrame, hotels: pd.DataFrame) -> pd.DataFrame:
    """
    Recurring calendar seasonality, isolated from the long-term decline
    by normalising each complete year (2020-2022) by its own mean.
    """
    f = flights.assign(year=flights["date"].dt.year,
                       month=flights["date"].dt.month)
    h = hotels.assign(year=hotels["date"].dt.year,
                      month=hotels["date"].dt.month)

    flight = (f[f["year"].isin(COMPLETE_YEARS)]
              .groupby(["year", "month"])["travelCode"].nunique()
              .rename("flight_trips").reset_index())
    hotel = (h[h["year"].isin(COMPLETE_YEARS)]
             .groupby(["year", "month"]).size()
             .rename("hotel_bookings").reset_index())

    for frame, value_col in [(flight, "flight_trips"),
                             (hotel, "hotel_bookings")]:
        frame["seasonal_index"] = frame.groupby("year")[value_col].transform(
            lambda s: s / s.mean())

    flight_season = flight.groupby("month")["seasonal_index"].mean()
    hotel_season = hotel.groupby("month")["seasonal_index"].mean()

    months = list(range(1, 13))
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(months, [flight_season.get(m, np.nan) for m in months],
            marker="o", label="Flight trips")
    ax.plot(months, [hotel_season.get(m, np.nan) for m in months],
            marker="s", label="Hotel bookings")
    ax.axhline(1.0, linestyle="--", linewidth=1, label="Annual average")
    ax.set_xticks(months)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("Seasonal index")
    ax.set_title("Seasonality of Travel Activity")
    ax.legend(title="Series", frameon=True)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save_fig(fig, "seasonality_index.png", dpi=180)

    seasonal = pd.DataFrame({
        "month": months,
        "flight_seasonal_index": [flight_season.get(m, np.nan) for m in months],
        "hotel_seasonal_index": [hotel_season.get(m, np.nan) for m in months],
    })
    save_table(seasonal, "seasonality_index.csv")
    return seasonal


def adaptive_pricing_exploration(flights: pd.DataFrame) -> pd.DataFrame:
    """
    Exploratory only. The data has no inventory, booking lead time,
    competitor price or cancellation columns, so this cannot support a
    causal dynamic-pricing claim.
    """
    monthly = (flights.assign(month=month_key(flights["date"]))
               .groupby("month")
               .agg(trips=("travelCode", "nunique"),
                    average_price=(TARGET, "mean"),
                    median_price=(TARGET, "median"))
               .reset_index())
    monthly["demand_change_pct"] = monthly["trips"].pct_change() * 100
    monthly["price_change_pct"] = monthly["average_price"].pct_change() * 100
    save_table(monthly, "adaptive_pricing_exploration.csv")

    annual = (flights.groupby(flights["date"].dt.year)
              .agg(trips=("travelCode", "nunique"),
                   active_users=("userCode", "nunique"),
                   average_price=(TARGET, "mean"),
                   median_price=(TARGET, "median"),
                   total_expenditure=(TARGET, "sum"))
              .rename_axis("year").reset_index())
    save_table(annual, "annual_activity_price_summary.csv")
    return monthly


def load_brazil_boundary(skip_geo: bool = False):
    """Natural Earth boundary, cached so repeat runs need no download."""
    if skip_geo:
        return None
    try:
        import geopandas as gpd
        from urllib.request import urlretrieve
    except ImportError:
        print("    geopandas unavailable - skipping basemap")
        return None

    GEO_CACHE_DIR.mkdir(exist_ok=True)
    zip_path = GEO_CACHE_DIR / "ne_110m_admin_0_countries.zip"
    url = ("https://naciscdn.org/naturalearth/110m/cultural/"
           "ne_110m_admin_0_countries.zip")

    if not zip_path.exists():
        print("    downloading Natural Earth map data...")
        try:
            urlretrieve(url, zip_path)
        except Exception as exc:
            print(f"    download failed ({exc}) - coordinate-only plot")
            return None

    try:
        world = gpd.read_file(f"zip://{zip_path}")
        return world[world["ADMIN"] == "Brazil"].to_crs("EPSG:4326")
    except Exception as exc:
        print(f"    could not read basemap ({exc}) - coordinate-only plot")
        return None


def plot_geographical_frequency(flights: pd.DataFrame,
                                hotels: pd.DataFrame) -> pd.DataFrame:
    flight_counts = pd.concat([
        flights["from"].value_counts(), flights["to"].value_counts()
    ]).groupby(level=0).sum()
    hotel_counts = hotels["place"].value_counts()

    places = sorted(set(flight_counts.index) | set(hotel_counts.index))
    geo = pd.DataFrame({"place": places})
    geo["flight_frequency"] = geo["place"].map(flight_counts).fillna(0)
    geo["hotel_frequency"] = geo["place"].map(hotel_counts).fillna(0)
    geo["total_frequency"] = geo["flight_frequency"] + geo["hotel_frequency"]

    geo["latitude"] = geo["place"].map(
        lambda x: CITY_COORDS.get(x, (np.nan, np.nan))[0])
    geo["longitude"] = geo["place"].map(
        lambda x: CITY_COORDS.get(x, (np.nan, np.nan))[1])
    geo = geo.dropna(subset=["latitude", "longitude"])

    fig, ax = plt.subplots(figsize=(10, 9))
    brazil = load_brazil_boundary()
    if brazil is not None:
        brazil.plot(ax=ax, facecolor="#eeeeee", edgecolor="black",
                    linewidth=0.8, zorder=1)

    scatter = ax.scatter(geo["longitude"], geo["latitude"],
                         s=np.maximum(geo["total_frequency"] / 180, 35),
                         c=geo["total_frequency"], cmap="YlOrRd",
                         alpha=0.80, edgecolor="black", linewidth=0.6,
                         zorder=3)
    for _, row in geo.iterrows():
        ax.annotate(row["place"], (row["longitude"], row["latitude"]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8, zorder=4)

    ax.set_title("Geographical Travel Frequency")
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.78, pad=0.02)
    cbar.set_label("Total observed travel frequency")

    sizes = sorted({max(int(v), 1) for v in geo["total_frequency"].quantile(
        [0.25, 0.50, 0.75])})
    if sizes:
        ax.legend(handles=[
            ax.scatter([], [], s=max(v / 180, 35), facecolor="gray",
                       edgecolor="black", alpha=0.6, label=f"{v:,}")
            for v in sizes
        ], title="Marker size = frequency", loc="lower left", frameon=True)

    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    save_fig(fig, "geographical_frequency.png", dpi=180)

    save_table(geo, "geographical_frequency.csv")
    return geo


# =====================================================================
# SECTION 2 - Hotel & user analysis
# =====================================================================

def plot_hotel_place_frequency(hotels: pd.DataFrame) -> None:
    places = hotels["place"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    places.plot(kind="barh", ax=ax)
    ax.set_title("Hotel Location Frequency")
    ax.set_xlabel("Number of hotel bookings")
    ax.set_ylabel("Place")
    fig.tight_layout()
    save_fig(fig, "hotel_place_frequency.png")


def plot_hotel_expenditure_distribution(hotels: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(hotels["total"], bins=30)
    ax.set_title("Hotel Total Expenditure Distribution")
    ax.set_xlabel("Hotel expenditure")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    save_fig(fig, "hotel_expenditure_distribution.png")


def plot_hotel_correlation(hotels: pd.DataFrame) -> None:
    corr = hotels[["days", "price", "total"]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(corr.values, aspect="auto", cmap="coolwarm",
                      vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.columns)
    ax.set_title("Hotel Numerical Feature Correlations")
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Pearson correlation")
    fig.tight_layout()
    save_fig(fig, "correlation_hotels.png")


def plot_user_demographics(users: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(users["age"], bins=15)
    ax.set_title("User Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of users")
    fig.tight_layout()
    save_fig(fig, "age_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    users["gender"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Gender Distribution")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Number of users")
    fig.tight_layout()
    save_fig(fig, "gender_distribution.png")


def build_user_expenditure(flights: pd.DataFrame,
                           hotels: pd.DataFrame) -> pd.DataFrame:
    flight_user = flights.groupby("userCode")[TARGET].sum().rename(
        "flight_expenditure")
    hotel_user = hotels.groupby("userCode")["total"].sum().rename(
        "hotel_expenditure")
    exp = pd.concat([flight_user, hotel_user], axis=1).fillna(0)
    exp["total_expenditure"] = exp["flight_expenditure"] + exp["hotel_expenditure"]
    return exp.reset_index()


def plot_user_expenditure(flights: pd.DataFrame,
                          hotels: pd.DataFrame) -> pd.DataFrame:
    user_exp = build_user_expenditure(flights, hotels)
    save_table(user_exp, "user_expenditure.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(user_exp["total_expenditure"], bins=30)
    ax.set_title("User Total Travel Expenditure")
    ax.set_xlabel("Total travel expenditure")
    ax.set_ylabel("Number of users")
    fig.tight_layout()
    save_fig(fig, "user_total_expenditure.png")

    # Flight class is the only available behavioural segmentation; the
    # dataset contains no business/pleasure purpose field.
    user_class = (flights.groupby(["userCode", "flightType"])[TARGET].sum()
                  .reset_index())
    merged = user_class.merge(user_exp[["userCode", "hotel_expenditure"]],
                              on="userCode", how="left")
    merged["combined_expenditure"] = merged[TARGET] + merged[
        "hotel_expenditure"].fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    for cls in merged["flightType"].unique():
        ax.hist(merged.loc[merged["flightType"] == cls, "combined_expenditure"],
                bins=25, alpha=0.5, label=cls)
    ax.set_title("User Expenditure by Flight Class")
    ax.set_xlabel("Travel expenditure")
    ax.set_ylabel("Number of user-class observations")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "user_expenditure_by_flight_class.png")
    return user_exp


def build_trip_level_data(flights: pd.DataFrame,
                          hotels: pd.DataFrame) -> pd.DataFrame:
    """One row per travelCode, joining flight and hotel spend."""
    flight_trip = (flights.groupby("travelCode", as_index=False)
                   .agg(flight_expenditure=(TARGET, "sum"),
                        flight_distance_total=("distance", "sum"),
                        flight_time_total=("time", "sum"),
                        flight_legs=(TARGET, "size"),
                        origin=("from", "first"),
                        destination=("to", "first"),
                        flight_type=("flightType", "first"),
                        agency=("agency", "first"),
                        userCode=("userCode", "first"),
                        date=("date", "min")))
    hotel_trip = (hotels.groupby("travelCode", as_index=False)
                  .agg(hotel_expenditure=("total", "sum"),
                       hotel_days=("days", "sum"),
                       hotel_name=("name", "first"),
                       hotel_place=("place", "first")))

    trip = flight_trip.merge(hotel_trip, on="travelCode", how="left")
    trip["hotel_expenditure"] = trip["hotel_expenditure"].fillna(0)
    trip["hotel_days"] = trip["hotel_days"].fillna(0)
    trip["total_trip_expenditure"] = (trip["flight_expenditure"]
                                      + trip["hotel_expenditure"])
    return add_date_features(trip)


def plot_trip_expenditure(flights: pd.DataFrame,
                          hotels: pd.DataFrame) -> pd.DataFrame:
    trip = build_trip_level_data(flights, hotels)
    save_table(trip, "trip_level_data.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(trip["total_trip_expenditure"], bins=30)
    ax.set_title("Trip-Level Total Expenditure")
    ax.set_xlabel("Flight + hotel expenditure")
    ax.set_ylabel("Number of trips")
    fig.tight_layout()
    save_fig(fig, "trip_expenditure_distribution.png")
    return trip


# =====================================================================
# SECTION 3 - Leakage analysis
# =====================================================================

def configuration_profile(
    df: pd.DataFrame,
    group_cols: list,
    target: str = TARGET,
    tol: float = 1e-9,
) -> tuple[dict, pd.DataFrame]:
    """
    Cardinality and target-variance profile for one column grouping.

    Returns (summary dict, per-configuration DataFrame).

    Metrics
    -------
    n_unique_configurations        distinct combinations of `group_cols`
    duplicated_configurations      configurations appearing more than once
    duplicated_rows                rows living in a duplicated configuration
    rows_per_configuration         mean / median / max replication
    within_config_price_variance   pooled variance of target inside configs
    between_config_price_variance  variance of per-config target means
    unexplained_variance_ratio     within / total sum of squares
    """
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise KeyError(f"columns not present in frame: {missing}")

    grouped = df.groupby(group_cols, observed=True)[target]
    per_config = grouped.agg(
        rows="size",
        price_nunique="nunique",
        price_mean="mean",
        price_std="std",
        price_min="min",
        price_max="max",
    ).reset_index()
    per_config["price_range"] = per_config["price_max"] - per_config["price_min"]
    per_config = per_config.sort_values("rows", ascending=False)

    n_rows = len(df)
    n_config = len(per_config)
    dup_mask = per_config["rows"] > 1

    total_ss = float(((df[target] - df[target].mean()) ** 2).sum())
    within_ss = float((per_config["price_std"].fillna(0.0)
                       ** 2 * per_config["rows"]).sum())
    n_deterministic = int((per_config["price_std"].fillna(0.0) <= tol).sum())

    summary = {
        "group_cols": " + ".join(group_cols),
        "n_rows": n_rows,
        "n_unique_configurations": n_config,
        "n_unique_prices": int(df[target].nunique()),
        "duplicated_configurations": int(dup_mask.sum()),
        "singleton_configurations": int((~dup_mask).sum()),
        "duplicated_rows": int(per_config.loc[dup_mask, "rows"].sum()),
        "rows_per_configuration_mean": round(
            float(per_config["rows"].mean()), 2),
        "rows_per_configuration_median": float(per_config["rows"].median()),
        "rows_per_configuration_max": int(per_config["rows"].max()),
        "redundancy_factor": round(n_rows / max(n_config, 1), 1),
        "configs_with_single_price": int((per_config["price_nunique"] == 1).sum()),
        "deterministic_configs": n_deterministic,
        "deterministic_config_pct": round(
            100.0 * n_deterministic / max(n_config, 1), 4),
        "max_within_config_price_std": float(
            per_config["price_std"].fillna(0.0).max()),
        "max_within_config_price_range": float(per_config["price_range"].max()),
        "within_config_price_variance": round(
            float(per_config["price_std"].fillna(0.0).pow(2).mul(
                per_config["rows"]).sum() / max(n_rows, 1)), 12),
        "between_config_price_variance": round(
            float(per_config["price_mean"].var(ddof=0)), 4),
        "total_price_variance": round(float(df[target].var(ddof=0)), 4),
        "unexplained_variance_ratio": round(within_ss / max(total_ss, 1e-12), 12),
        "lookup_R2": round(1.0 - within_ss / max(total_ss, 1e-12), 12),
        "price_is_deterministic": bool(n_deterministic == n_config),
    }
    return summary, per_config


def split_configuration_overlap(train: pd.DataFrame, test: pd.DataFrame,
                                group_cols: list) -> dict:
    """How many test configurations were already memorised in training."""
    train_keys = set(map(tuple, train[group_cols].to_numpy()))
    test_keys = pd.Series(list(map(tuple, test[group_cols].to_numpy())))
    seen = test_keys.isin(train_keys)
    return {
        "group_cols": " + ".join(group_cols),
        "n_train_configs": len(train_keys),
        "n_test_configs": int(test_keys.nunique()),
        "test_configs_seen_in_train": int(
            test_keys[seen].nunique()),
        "test_rows_on_seen_configs_pct": round(100.0 * float(seen.mean()), 4),
        "test_rows_on_unseen_configs": int((~seen).sum()),
    }


def find_minimal_deterministic_key(
    df: pd.DataFrame,
    candidates: list,
    target: str = TARGET,
    tol: float = 1e-9,
) -> tuple[list, pd.DataFrame]:
    """
    Greedily shrink the feature set to the smallest subset that still
    determines the target exactly.

    Shows how much of the reported model complexity is redundant.
    """
    key: list = []
    remaining = list(candidates)
    trace = []

    while remaining:
        best_col, best_ratio, best_configs = None, np.inf, np.inf
        for col in remaining:
            trial = key + [col]
            g = df.groupby(trial, observed=True)[target]
            per = g.agg(rows="size", std="std").reset_index()
            within_ss = float((per["std"].fillna(0.0) ** 2 * per["rows"]).sum())
            total_ss = float(((df[target] - df[target].mean()) ** 2).sum())
            ratio = within_ss / max(total_ss, 1e-12)
            if ratio < best_ratio:
                best_col, best_ratio, best_configs = col, ratio, len(per)

        if best_ratio >= 1.0 - 1e-12:
            break  # cannot do better

        key = key + [best_col]
        remaining.remove(best_col)
        trace.append({
            "step": len(key),
            "added_column": best_col,
            "key": " + ".join(key),
            "n_configurations": int(best_configs),
            "redundancy_factor": round(len(df) / max(best_configs, 1), 1),
            "unexplained_variance_ratio": best_ratio,
            "is_deterministic": bool(best_ratio <= tol),
        })
        if best_ratio <= tol:
            break

    return key, pd.DataFrame(trace)


def plot_configuration_cardinality(summaries: list[dict]) -> None:
    labels = ["with date\n(9 columns)", "no date\n(6 columns)"]
    n_config = [s["n_unique_configurations"] for s in summaries]
    n_rows = [s["n_rows"] for s in summaries]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2, n_rows, width, label="Rows in dataset")
    ax.bar(x + width / 2, n_config, width, label="Unique configurations")
    for i, (r, c) in enumerate(zip(n_rows, n_config)):
        ax.annotate(f"{r:,}", (i - width / 2, r), ha="center", va="bottom")
        ax.annotate(f"{c:,}", (i + width / 2, c), ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Row Count vs Unique Feature Configurations")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "configuration_cardinality.png")


def plot_unique_price_levels(frame: pd.DataFrame) -> None:
    """
    Visual proof that the target has only a few hundred distinct values.

    Three complementary views, because each one alone is easy to
    misread on a 272k-row frame:
      (a) every distinct price as a sorted stem plot - shows the exact
          discrete grid the model can ever emit,
      (b) a histogram binned at one bin per distinct price - the spikes
          align 1:1 with the grid,
      (c) a zoom on the busiest price levels with their row counts.
    """
    prices = np.sort(frame[TARGET].unique())
    counts = frame[TARGET].value_counts()

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    ax = axes[0]
    ax.vlines(np.arange(len(prices)), 0, prices, color="#4C72B0",
              linewidth=0.6, alpha=0.75)
    ax.plot(np.arange(len(prices)), prices, linestyle="none", marker="o",
            markersize=2.5, color="#C44E52")
    ax.set_title(f"All {len(prices):,} distinct price values\n"
                 f"({len(frame):,} rows -> {len(prices):,} levels, "
                 f"{len(frame)/len(prices):.0f}x redundancy)")
    ax.set_xlabel("Sorted distinct price index")
    ax.set_ylabel("Flight price")

    ax = axes[1]
    ax.hist(frame[TARGET], bins=len(prices))
    ax.set_yscale("log")
    ax.set_title(f"Price histogram, one bin per distinct value\n"
                 f"({len(prices)} bins)")
    ax.set_xlabel("Flight price")
    ax.set_ylabel("Rows (log scale)")

    ax = axes[2]
    top = counts.head(25).sort_values()
    ax.barh([f"{v:,.2f}" for v in top.index], top.to_numpy(),
            color="#55A868")
    ax.set_title("25 most frequent price values\n"
                 "each repeated hundreds of times")
    ax.set_xlabel("Rows")
    ax.set_ylabel("Price")
    ax.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    save_fig(fig, "unique_price_levels.png")


def plot_duplication_structure(frame: pd.DataFrame,
                               per_config: pd.DataFrame,
                               summaries: list[dict]) -> None:
    """
    Quantify the duplication: how many rows collapse into each
    configuration, and how that compares across the two groupings.
    """
    sizes = per_config["rows"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    ax = axes[0]
    ax.hist(sizes, bins=60, color="#4C72B0")
    ax.axvline(sizes.mean(), color="#C44E52", linestyle="--",
               label=f"mean {sizes.mean():.0f}")
    ax.axvline(np.median(sizes), color="#55A868", linestyle=":",
               label=f"median {np.median(sizes):.0f}")
    ax.set_title("Rows per 6-column configuration\n"
                 f"mean {sizes.mean():.0f}, max {sizes.max():,.0f}")
    ax.set_xlabel("Rows sharing one configuration")
    ax.set_ylabel("Configurations")
    ax.legend()

    ax = axes[1]
    ax.hist(sizes, bins=60, color="#4C72B0")
    ax.set_yscale("log")
    ax.axvline(sizes.mean(), color="#C44E52", linestyle="--",
               label=f"mean {sizes.mean():.0f}")
    ax.set_title("Same, log y-scale\n(long tail of large groups)")
    ax.set_xlabel("Rows sharing one configuration")
    ax.set_ylabel("Configurations (log scale)")
    ax.legend()

    ax = axes[2]
    labels = ["with date\n(9 cols)", "no date\n(6 cols)"]
    x = np.arange(len(labels))
    width = 0.26
    ax.bar(x - width, [s["n_rows"] for s in summaries], width,
           label="Rows", color="#4C72B0")
    ax.bar(x, [s["n_unique_configurations"] for s in summaries], width,
           label="Configurations", color="#DD8452")
    ax.bar(x + width, [s["n_unique_prices"] for s in summaries], width,
           label="Distinct prices", color="#C44E52")
    for i, s in enumerate(summaries):
        ax.annotate(f"{s['redundancy_factor']}x", (i, s["n_rows"]),
                    ha="center", va="bottom", fontsize=9,
                    color="#C44E52", fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Dropping the date columns collapses 272k rows\n"
                 "onto 490 configurations = 490 prices")
    ax.legend()

    fig.tight_layout()
    save_fig(fig, "duplication_structure.png")


def plot_within_config_variance(per_configs: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (label, per) in zip(axes, per_configs.items()):
        ranges = per["price_range"].to_numpy(dtype=float)
        ax.hist(np.clip(ranges, 0, 1e-6), bins=50, log=True)
        ax.set_title(f"Within-configuration price range\n{label}")
        ax.set_xlabel("Price range inside configuration (clipped at 1e-6)")
        ax.set_ylabel("Configurations (log scale)")
    fig.tight_layout()
    save_fig(fig, "within_configuration_price_variance.png")


def leakage_analysis(frame: pd.DataFrame) -> dict:
    """
    Dedicated leakage section.

    Quantifies how much of the target is explained by configuration
    identity alone, then compares what a random split reports against
    what a route-held-out split reports.
    """
    banner("SECTION 3 - LEAKAGE ANALYSIS")

    groupings = [
        ("with date (9 columns)", GROUP_COLS_WITH_DATE),
        ("no date (6 columns)", GROUP_COLS_NO_DATE),
    ]

    summaries, per_config_frames = [], {}
    for label, cols in groupings:
        print(f"\n  [{label}]")
        summary, per_config = configuration_profile(frame, cols)
        summaries.append(summary)
        per_config_frames[label] = per_config

        suffix = "with_date" if "with date" in label else "no_date"
        save_table(
            per_config.head(2000),
            f"{suffix}_configuration_stats.csv",
        )
        print(f"    unique configurations : {summary['n_unique_configurations']:,}")
        print(f"    duplicated configs    : "
              f"{summary['duplicated_configurations']:,}")
        print(f"    duplicated rows       : {summary['duplicated_rows']:,}")
        print(f"    rows per config       : "
              f"mean {summary['rows_per_configuration_mean']}, "
              f"median {summary['rows_per_configuration_median']}, "
              f"max {summary['rows_per_configuration_max']}")
        print(f"    unique prices         : {summary['n_unique_prices']:,}")
        print(f"    deterministic configs : "
              f"{summary['deterministic_configs']:,}/"
              f"{summary['n_unique_configurations']:,} "
              f"({summary['deterministic_config_pct']}%)")
        print(f"    max within-config std : "
              f"{summary['max_within_config_price_std']:.3e}")
        print(f"    unexplained var ratio : "
              f"{summary['unexplained_variance_ratio']:.3e}")
        print(f"    price deterministic   : "
              f"{summary['price_is_deterministic']}")

    summary_df = pd.DataFrame([
        {k: v for k, v in s.items()} for s in summaries
    ])
    save_table(summary_df, "configuration_cardinality_summary.csv")
    plot_configuration_cardinality(summaries)
    plot_within_config_variance(per_config_frames)

    # ---- minimal key that still determines price exactly ------------
    print("\n  [minimal deterministic key search]")
    key, trace = find_minimal_deterministic_key(
        frame, GROUP_COLS_WITH_DATE
    )
    if not trace.empty:
        save_table(trace, "minimal_deterministic_key.csv")
        for _, r in trace.iterrows():
            print(f"    +{r['added_column']:<12s} -> {r['n_configurations']:,} "
                  f"configs, residual {r['unexplained_variance_ratio']:.3e}"
                  f"{'  [DETERMINISTIC]' if r['is_deterministic'] else ''}")

    # ---- what a random split memorises ------------------------------
    print("\n  [split configuration overlap]")
    overlap_rows = []
    tr_idx, te_idx = train_test_split(
        np.arange(len(frame)), test_size=TEST_SIZE, random_state=SEED
    )
    train, test = frame.iloc[tr_idx], frame.iloc[te_idx]
    for label, cols in groupings:
        info = split_configuration_overlap(train, test, cols)
        overlap_rows.append(info)
        print(f"    {label:<22s} test configs seen in train: "
              f"{info['test_configs_seen_in_train']:,}/"
              f"{info['n_test_configs']:,}; "
              f"rows on unseen configs: {info['test_rows_on_unseen_configs']:,}")
    overlap_df = pd.DataFrame(overlap_rows)
    save_table(overlap_df, "split_configuration_overlap.csv")

    # ---- duplicate-record audit -------------------------------------
    print("\n  [duplicate row audit]")
    dup_rows = frame.duplicated().sum()
    dup_configs = int(
        (frame.groupby(GROUP_COLS_NO_DATE, observed=True)
         .size() > 1).sum()
    )
    print(f"    fully duplicated rows              : {dup_rows:,}")
    print(f"    duplicated 6-column configurations : {dup_configs:,}")
    print("    -> duplicated rows are distinct travelCode/date records "
          "sharing an identical fare")

    # ---- explicit evidence tables + figures -------------------------
    print("\n  [distinct price values]")
    price_table = (frame[TARGET].value_counts()
                   .rename_axis("price")
                   .reset_index(name="rows"))
    price_table["share_pct"] = (price_table["rows"] / len(frame) * 100).round(4)
    price_table = price_table.sort_values("price").reset_index(drop=True)
    save_table(price_table, "unique_price_values.csv")
    print(f"    distinct price values : {len(price_table):,}")
    print(f"    rows per value        : mean "
          f"{price_table['rows'].mean():.1f}, "
          f"min {int(price_table['rows'].min())}, "
          f"max {int(price_table['rows'].max())}")
    print(f"    top value covers      : "
          f"{price_table['rows'].max() / len(frame) * 100:.2f}% of all rows")

    no_date_summary = summaries[-1]
    plot_unique_price_levels(frame)
    plot_duplication_structure(frame, per_config_frames[
        "no date (6 columns)"], summaries)

    report = {
        "cardinality": summaries,
        "minimal_deterministic_key": key,
        "minimal_key_trace": trace.to_dict("records"),
        "split_overlap": overlap_df.to_dict("records"),
        "n_unique_prices": int(len(price_table)),
        "price_values": price_table.to_dict("records"),
        "duplicate_rows": int(dup_rows),
        "duplicated_configurations_no_date": dup_configs,
        "redundancy_factor_no_date": no_date_summary["redundancy_factor"],
    }
    (OUTPUT_DIR / "leakage_analysis.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print("    saved leakage_analysis.json")

    return report


# =====================================================================
# SECTION 4 - ML training under two regimes
# =====================================================================
#
# The same three regressors are trained twice:
#
#   REGIME A - randomized split, exactly as the original notebook did it.
#              LabelEncoder fitted on the FULL dataset before splitting,
#              StandardScaler for the linear model, train_test_split
#              (test_size=0.30, random_state=42), single 70/30 split.
#
#   REGIME B - GroupKFold on `from`, leakage prevented.
#              Preprocessing is fitted INSIDE each fold, and every
#              validation fold holds out whole origin cities, so no test
#              route was ever seen during training.
#
# The point of the comparison is that the only meaningful difference is
# the split strategy. Anything else (encoder type, scaler) is reported so
# the comparison is not silently confounded.
#
# Lookup tables are deliberately excluded: the exercise is to compare
# models, not to short-circuit to a table.

RANDOM_REGIME = "random_split"
GROUPED_REGIME = "groupkfold"
REGIME_A_LABEL = "A: random split (original)"
REGIME_B_LABEL = "B: GroupKFold (leakage-free)"


def model_factories(n_estimators: int = 100) -> dict:
    """
    The single source of truth for which models are trained.

    Both regimes iterate over this dict, so a model can never appear in
    one regime and be missing from the other.
    """
    return {
        "LinearRegression": lambda: LinearRegression(),
        "DecisionTree": lambda: DecisionTreeRegressor(random_state=SEED),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=n_estimators, random_state=SEED, n_jobs=-1),
    }


def original_style_preprocessing(frame: pd.DataFrame) -> tuple:
    """
    Reproduce the notebook's preprocessing exactly, including its flaws.

    The notebook called ``LabelEncoder.fit_transform`` on the whole
    dataframe before ``train_test_split``, so the category vocabulary was
    derived from test rows too. Reproduced deliberately so the published
    numbers can be matched, not because it is correct.
    """
    encoded = frame.copy()
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        enc = LabelEncoder()
        encoded[col] = enc.fit_transform(encoded[col])
        encoders[col] = enc
    return encoded, encoders


def leakage_free_preprocessor() -> ColumnTransformer:
    """
    Preprocessor for regime B.

    One-hot encoding removes the false ordering LabelEncoder imposes on
    `from` / `to`, and ``handle_unknown='ignore'`` lets the model score
    cities absent from a fold instead of raising. Because this is built
    fresh per fold, no test information reaches the encoder.
    """
    return ColumnTransformer(
        transformers=[(
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            CATEGORICAL_FEATURES,
        )],
        remainder="passthrough",
        sparse_threshold=1.0,
    )


def grouped_cv_preprocessor() -> ColumnTransformer:
    """
    Regime B preprocessor with numeric scaling for the linear model.

    StandardScaler on the numeric block only. Trees are invariant to it,
    so one shared preprocessor keeps the two regimes comparable: the only
    thing that changes between a linear and a tree model is the estimator.
    """
    return ColumnTransformer(
        transformers=[
            ("categorical",
             OneHotEncoder(handle_unknown="ignore", sparse_output=True),
             CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def score_predictions(y_true, y_pred, **extra) -> dict:
    """Attach the shared metric set plus any per-call context."""
    row = dict(extra)
    row.update(regression_metrics(y_true, y_pred))
    return row


def train_random_split_regime(frame: pd.DataFrame) -> tuple:
    """
    REGIME A - train exactly as the original notebook did.

    LabelEncoder on the full dataset, StandardScaler for the linear model
    fitted on the training fold only, one random 70/30 row split. This is
    the configuration that produced the published MAE 0.0845 / R2 0.99999.
    """
    banner("SECTION 4.1 - REGIME A: randomized split (original method)")

    encoded, _ = original_style_preprocessing(frame)
    X = encoded[MODEL_FEATURES]
    y = encoded[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED)
    print(f"\n  train rows {len(X_train):,} | test rows {len(X_test):,}")
    print(f"  encoders fitted on the full dataset before splitting "
          f"({len(frame):,} rows) - regime A keeps this flaw on purpose")

    # How much memorisation does this split hand the model?
    train_keys = set(map(tuple, X_train[GROUP_COLS_NO_DATE].to_numpy()))
    test_keys = pd.Series(list(map(tuple, X_test[GROUP_COLS_NO_DATE].to_numpy())))
    seen = test_keys.isin(train_keys)
    print(f"  test rows whose 6-column configuration appears in train: "
          f"{seen.mean() * 100:.2f}%")

    # StandardScaler mirrors the notebook (which duplicated the fit call).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results, predictions = [], {}
    for name, factory in model_factories().items():
        model = factory()
        started = time.time()
        if name == "LinearRegression":
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        metrics = regression_metrics(y_test, pred)
        print(f"    {name:<20s} MAE {metrics['MAE']:10.4f} | "
              f"RMSE {metrics['RMSE']:10.4f} | R2 {metrics['R2']:.6f} | "
              f"MAPE {metrics['MAPE_pct']:6.2f}%")
        results.append({
            "Model": name, "Regime": RANDOM_REGIME, "Fold": "holdout",
            **metrics,
            "pct_test_rows_seen_in_train": round(float(seen.mean()) * 100, 2),
            "fit_seconds": round(time.time() - started, 2),
        })
        predictions[name] = (y_test.to_numpy(), pred)
        del model
        gc.collect()

    results_df = pd.DataFrame(results)
    save_table(results_df, "ml_random_split_metrics.csv")
    return results_df, predictions


def train_groupkfold_regime(frame: pd.DataFrame, n_splits: int = 5) -> tuple:
    """
    REGIME B - GroupKFold on the origin city, leakage prevented.

    Two guarantees the randomized split does not provide:
      1. every encoder/scaler is fitted inside its own training fold, so
         no test row informs the preprocessing;
      2. each validation fold contains whole origin cities, so no test
         route was observed during training.

    Metrics come from out-of-fold predictions over the entire dataset,
    which is a full-coverage unbiased estimate rather than a 30% sample.
    """
    banner(f"SECTION 4.2 - REGIME B: GroupKFold on origin "
           f"({n_splits} folds, leakage prevented)")

    X = frame[MODEL_FEATURES]
    y = frame[TARGET]
    groups = frame["from"]

    n_splits = max(2, min(n_splits, int(groups.nunique())))
    cv = GroupKFold(n_splits=n_splits)
    splits = list(cv.split(X, y, groups=groups))
    print(f"\n  folds: {len(splits)} | rows: {len(frame):,} | "
          f"origin cities: {groups.nunique()}")
    for i, (tr, te) in enumerate(splits, 1):
        held = sorted(groups.iloc[te].unique())
        print(f"    fold {i}: train {len(tr):>7,} | test {len(te):>7,} | "
              f"held-out origins: {', '.join(c.split(' (')[0] for c in held)}")

    # Confirm the grouping actually removes route overlap.
    for i, (tr, te) in enumerate(splits, 1):
        tr_keys = set(map(tuple, frame.iloc[tr][GROUP_COLS_NO_DATE].to_numpy()))
        te_keys = pd.Series(list(map(
            tuple, frame.iloc[te][GROUP_COLS_NO_DATE].to_numpy())))
        leaked = float(te_keys.isin(tr_keys).mean()) * 100
        print(f"    fold {i}: test rows with a configuration seen in train: "
              f"{leaked:.4f}%")

    results, per_fold, predictions = [], [], {}
    for name, factory in model_factories().items():
        oof = np.full(len(frame), np.nan)
        fold_metrics = []
        started = time.time()

        for fold, (tr, te) in enumerate(splits, 1):
            pipe = Pipeline([
                ("prep", grouped_cv_preprocessor()),
                ("model", factory()),
            ])
            pipe.fit(X.iloc[tr], y.iloc[tr])
            pred = pipe.predict(X.iloc[te])
            oof[te] = pred
            m = regression_metrics(y.iloc[te], pred)
            fold_metrics.append(m)
            per_fold.append({
                "Model": name, "Regime": GROUPED_REGIME, "Fold": fold,
                "n_train_rows": int(len(tr)), "n_test_rows": int(len(te)),
                "held_out_origins": ",".join(sorted(groups.iloc[te].unique())),
                **m,
            })
            del pipe, pred
            gc.collect()

        pooled = regression_metrics(y.to_numpy(), oof)
        print(f"\n    {name}")
        print(f"      out-of-fold MAE   {pooled['MAE']:9.4f} | "
              f"RMSE {pooled['RMSE']:9.4f} | R2 {pooled['R2']:.6f}")
        print(f"      per-fold MAE      "
              f"{[round(f['MAE'], 2) for f in fold_metrics]}")
        print(f"      per-fold R2       "
              f"{[round(f['R2'], 3) for f in fold_metrics]}")
        print(f"      fold MAE spread   "
              f"{np.mean([f['MAE'] for f in fold_metrics]):.2f} "
              f"+/- {np.std([f['MAE'] for f in fold_metrics]):.2f}")

        results.append({
            "Model": name, "Regime": GROUPED_REGIME, "Fold": "out_of_fold",
            **pooled,
            "fold_MAE_mean": round(
                float(np.mean([f["MAE"] for f in fold_metrics])), 4),
            "fold_MAE_std": round(
                float(np.std([f["MAE"] for f in fold_metrics])), 4),
            "fold_MAE_min": round(
                float(np.min([f["MAE"] for f in fold_metrics])), 4),
            "fold_MAE_max": round(
                float(np.max([f["MAE"] for f in fold_metrics])), 4),
            "fold_R2_min": round(
                float(np.min([f["R2"] for f in fold_metrics])), 6),
            "fold_R2_max": round(
                float(np.max([f["R2"] for f in fold_metrics])), 6),
            "pct_test_rows_seen_in_train": 0.0,
            "fit_seconds": round(time.time() - started, 2),
        })
        predictions[name] = (y.to_numpy(), oof)
        gc.collect()

    results_df = pd.DataFrame(results)
    folds_df = pd.DataFrame(per_fold)
    save_table(results_df, "ml_groupkfold_metrics.csv")
    save_table(folds_df, "ml_groupkfold_per_fold.csv")
    return results_df, folds_df, predictions


def regime_comparison(random_res: pd.DataFrame,
                      grouped_res: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side view of the two regimes, with the gap made explicit."""
    left = random_res.rename(columns={
        "MAE": "MAE_random", "RMSE": "RMSE_random", "R2": "R2_random"})
    right = grouped_res.rename(columns={
        "MAE": "MAE_groupkfold", "RMSE": "RMSE_groupkfold",
        "R2": "R2_groupkfold"})

    merged = left[["Model", "MAE_random", "RMSE_random", "R2_random"]].merge(
        right[["Model", "MAE_groupkfold", "RMSE_groupkfold", "R2_groupkfold"]],
        on="Model")
    merged["MAE_inflation_factor"] = (
        merged["MAE_groupkfold"] / merged["MAE_random"].replace(0, np.nan)
    ).round(1)
    merged["R2_drop"] = (merged["R2_random"] - merged["R2_groupkfold"]).round(6)
    save_table(merged, "ml_regime_comparison.csv")
    return merged


def plot_actual_vs_predicted_regime(predictions: dict, regime: str,
                                    filename: str, title: str) -> None:
    """
    One row per model: actual-vs-predicted scatter plus residual spread.

    This is where the memorisation shows up visually - regime A produces
    near-perfect diagonal scatter, regime B produces a visibly widened
    band around it.
    """
    n_models = len(predictions)
    fig, axes = plt.subplots(n_models, 2, figsize=(13, 4.2 * n_models),
                             squeeze=False)

    for row, (name, (y_true, y_pred)) in enumerate(predictions.items()):
        ax = axes[row][0]
        ax.scatter(y_true, y_pred, alpha=0.12, s=6)
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black",
                linewidth=1)
        metrics = regression_metrics(y_true, y_pred)
        ax.set_title(f"{name}\nMAE {metrics['MAE']:.2f} | "
                     f"R2 {metrics['R2']:.4f}", fontsize=10)
        ax.set_xlabel("Actual price")
        ax.set_ylabel("Predicted price")

        ax = axes[row][1]
        resid = y_true - y_pred
        ax.hist(resid, bins=60, color="#4C72B0")
        ax.axvline(0, linestyle="--", color="black", linewidth=1)
        ax.set_title(f"{name} residuals\nmean {resid.mean():+.2f} | "
                     f"sd {resid.std():.2f} | "
                     f"|r|>100: {(np.abs(resid) > 100).mean() * 100:.1f}%",
                     fontsize=10)
        ax.set_xlabel("Actual - predicted")
        ax.set_ylabel("Rows")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save_fig(fig, filename)


def plot_metric_bar_comparison(comparison: pd.DataFrame) -> None:
    """
    Grouped bar charts comparing every model under both regimes.

    MAE and RMSE share a figure because they have the same units; R2 gets
    its own axis because it is dimensionless and would flatten the others.
    """
    models = comparison["Model"].tolist()
    x = np.arange(len(models))
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    for ax, metric, logscale, title in [
        (axes[0], "MAE", True, "Mean Absolute Error"),
        (axes[1], "RMSE", True, "Root Mean Squared Error"),
        (axes[2], "R2", False, "R-squared"),
    ]:
        a_vals = comparison[f"{metric}_random"].to_numpy(dtype=float)
        b_vals = comparison[f"{metric}_groupkfold"].to_numpy(dtype=float)

        ax.bar(x - width / 2, a_vals, width,
               label=REGIME_A_LABEL, color="#4C72B0")
        ax.bar(x + width / 2, b_vals, width,
               label=REGIME_B_LABEL, color="#C44E52")

        for i, (a, b) in enumerate(zip(a_vals, b_vals)):
            fmt = (lambda v: f"{v:,.0f}") if metric != "R2" else (
                lambda v: f"{v:.4f}")
            ax.annotate(fmt(a), (i - width / 2, a), ha="center",
                        va="bottom", fontsize=8)
            ax.annotate(fmt(b), (i + width / 2, b), ha="center",
                        va="bottom", fontsize=8)

        if logscale:
            # Regimes A and B differ by ~3 orders of magnitude, so a
            # linear axis would render the random-split bars invisible.
            ax.set_yscale("log")
            ax.set_ylabel(f"{metric} (log scale)")
        else:
            ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9)
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle("Same models, two split strategies: random split vs "
                 "GroupKFold on origin city", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, "ml_metric_bar_comparison.png")


def plot_fold_spread(per_fold: pd.DataFrame) -> None:
    """Per-fold MAE spread: shows how unstable the honest estimate is."""
    models = per_fold["Model"].drop_duplicates().tolist()
    folds = sorted(per_fold["Fold"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    width = 0.8 / len(models)
    x = np.arange(len(folds))
    for i, name in enumerate(models):
        subset = per_fold[per_fold["Model"] == name].set_index("Fold")
        vals = [subset.loc[f, "MAE"] for f in folds]
        ax.bar(x + i * width, vals, width, label=name)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([f"fold {f}" for f in folds])
    ax.set_ylabel("MAE")
    ax.set_title("Per-fold MAE (GroupKFold)")
    ax.legend()

    ax = axes[1]
    for i, name in enumerate(models):
        subset = per_fold[per_fold["Model"] == name].set_index("Fold")
        ax.plot(folds, [subset.loc[f, "R2"] for f in folds], marker="o",
                label=name)
    ax.axhline(0, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Fold")
    ax.set_ylabel("R2")
    ax.set_title("Per-fold R2 spread\n(regime A reports a single number)")
    ax.legend()

    fig.tight_layout()
    save_fig(fig, "ml_groupkfold_fold_spread.png")


def feature_importance_report(frame: pd.DataFrame,
                             perm_sample: int = 20_000) -> pd.DataFrame:
    """Permutation importance under the leakage-free regime."""
    banner("SECTION 4.3 - Permutation importance (GroupKFold regime)")

    X = frame[MODEL_FEATURES]
    y = frame[TARGET]
    groups = frame["from"]

    n_splits = max(2, min(5, int(groups.nunique())))
    splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))

    pipe = Pipeline([("prep", grouped_cv_preprocessor()),
                     ("model", model_factories()["RandomForest"]())])
    pipe.fit(X.iloc[splits[0][0]], y.iloc[splits[0][0]])

    from sklearn.inspection import permutation_importance

    # Importance is reported per INPUT column (the pipeline receives the 9
    # raw features), so no one-hot expansion mapping is needed here.
    #
    # The evaluation fold is subsampled and run single-process on purpose.
    # permutation_importance already has a fitted estimator, so there is no
    # fitting to parallelise; forking N workers each holding a 100-tree
    # forest over 270k rows exhausts memory on a 16 GB machine.
    rng = np.random.default_rng(SEED)
    eval_idx = splits[0][1]
    if len(eval_idx) > perm_sample:
        eval_idx = rng.choice(eval_idx, size=perm_sample, replace=False)
    result = permutation_importance(
        pipe, X.iloc[eval_idx], y.iloc[eval_idx], n_repeats=5,
        random_state=SEED, scoring="neg_mean_absolute_error", n_jobs=1,
    )

    importance = pd.DataFrame({
        "Feature": list(X.columns),
        "importance_mae": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mae", ascending=False)
    importance["importance_mae"] = importance["importance_mae"].round(4)
    importance["importance_std"] = importance["importance_std"].round(4)
    save_table(importance, "ml_permutation_importance.csv")

    print("\n  MAE increase when a feature is permuted "
          "(leakage-free validation fold):")
    for _, r in importance.iterrows():
        bar = "#" * int(min(abs(r["importance_mae"]), 60))
        print(f"    {r['Feature']:<12s} {r['importance_mae']:9.3f}  {bar}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(importance["Feature"], importance["importance_mae"])
    ax.set_xlabel("Increase in MAE when permuted")
    ax.set_ylabel("Feature")
    ax.set_title("Permutation Importance (leakage-free)")
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, "ml_permutation_importance.png")

    del pipe
    gc.collect()
    return importance


def save_deployable_model(frame: pd.DataFrame) -> Path | None:
    """
    Persist the selected pipeline as ONE artifact.

    The original project shipped five separate joblib files with a
    hand-maintained column order in app.py. A fitted Pipeline carries its
    own column contract, so the app cannot silently mis-align inputs.
    """
    banner("SECTION 4.4 - Artifact export")

    pipe = Pipeline([
        ("prep", grouped_cv_preprocessor()),
        ("model", model_factories()["RandomForest"]()),
    ])
    pipe.fit(frame[MODEL_FEATURES], frame[TARGET])

    path = ARTIFACT_DIR / "flight_price_pipeline.joblib"
    import joblib
    joblib.dump(pipe, path)
    size_mb = path.stat().st_size / 1e6

    manifest = {
        "sklearn_version": __import__("sklearn").__version__,
        "features": MODEL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "n_train_rows": int(len(frame)),
        "target": TARGET,
        "seed": SEED,
        "trained_under": (
            "fitted on the full dataset for serving; performance must be "
            "quoted from the GroupKFold regime, not a random split"
        ),
        "observed_ranges": {
            col: {"min": float(frame[col].min()),
                  "max": float(frame[col].max())}
            for col in NUMERIC_FEATURES
        },
        "observed_categories": {
            col: sorted(frame[col].unique().tolist())
            for col in CATEGORICAL_FEATURES
        },
        "observed_routes": sorted(
            frame.groupby(["from", "to"]).size().index.tolist()
        ),
        "warning": (
            "Configuration analysis shows price is a deterministic function "
            "of (from, to, flightType, time, distance, agency): 490 "
            "configurations, 490 distinct prices, zero within-group "
            "variance. This model interpolates known routes; under "
            "GroupKFold on origin it scores R2 ~0.55-0.85, not ~1.0."
        ),
    }
    (ARTIFACT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"    saved {path.name}  ({size_mb:.2f} MB)")
    print("    saved manifest.json")
    print("    single artifact replaces 5 joblib files "
          "(model + 4 LabelEncoders)")

    reloaded = joblib.load(path)
    sample = frame[MODEL_FEATURES].head(5)
    if not np.allclose(reloaded.predict(sample), pipe.predict(sample)):
        raise RuntimeError("exported pipeline does not reproduce predictions")
    print("    round-trip prediction check passed")

    del pipe
    gc.collect()
    return path


# =====================================================================
# SECTION 5 - Orchestration
# =====================================================================

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-rows", type=int, default=None,
                   help="truncate each CSV (fast smoke run)")
    p.add_argument("--skip-geo", action="store_true",
                   help="skip Brazil basemap download/render")
    p.add_argument("--cv-folds", type=int, default=5,
                   help="GroupKFold folds for cross-validation")
    p.add_argument("--no-save-model", action="store_true",
                   help="evaluate only; do not write ml_artifacts")
    p.add_argument("--max-threads", type=int, default=8,
                   help="cap BLAS/OMP threads to bound peak memory")
    p.add_argument("--perm-sample", type=int, default=20_000,
                   help="rows sampled for permutation importance")
    p.add_argument("--ml-max-rows", type=int, default=None,
                   help="subsample rows for the ML sections only "
                        "(leakage analysis always uses every row)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    warnings.filterwarnings("ignore", category=FutureWarning)
    ensure_dirs()

    # Each joblib worker holds a copy of the expanded design matrix.
    # Capping threads keeps peak memory predictable on modest machines.
    if args.max_threads:
        os.environ["OMP_NUM_THREADS"] = str(args.max_threads)
        os.environ["MKL_NUM_THREADS"] = str(args.max_threads)

    print("Voyage Analytics - merged EDA + ML + leakage analysis")
    print(f"  data   : {DATA_DIR}")
    print(f"  outputs: {OUTPUT_DIR}")
    print(f"  models : {ARTIFACT_DIR}")
    print(f"  seed   : {SEED}")

    banner("SECTION 0 - Load and clean")
    frames = load_data(max_rows=args.max_rows)
    for name, df in frames.items():
        print(f"    {name:<8s} {df.shape[0]:>9,} rows x {df.shape[1]:>2d} cols")

    users = clean_dataframe(frames["users"])
    hotels = clean_dataframe(frames["hotels"], date_columns=["date"])
    flights = clean_dataframe(frames["flights"], date_columns=["date"])
    flights = add_date_features(flights)
    hotels = add_date_features(hotels)

    quality = flight_data_quality(flights, hotels)
    print("\n  Missing values after cleaning:")
    print(quality.groupby("dataset")["missing"].sum().to_string())

    model_frame = build_model_frame(flights)
    leakage_frame = model_frame  # always the full dataset

    ml_frame = model_frame
    if args.ml_max_rows and args.ml_max_rows < len(model_frame):
        rng = np.random.default_rng(SEED)
        keep = rng.choice(len(model_frame), size=args.ml_max_rows,
                          replace=False)
        ml_frame = model_frame.iloc[np.sort(keep)].reset_index(drop=True)
        print(f"  ML sections subsampled to {len(ml_frame):,} rows "
              f"(--ml-max-rows {args.ml_max_rows:,})")
    print(f"\n  Model frame: {model_frame.shape[0]:,} rows x "
          f"{model_frame.shape[1]} cols ({MODEL_FEATURES} -> {TARGET})")

    banner("SECTION 1 - Flight data analysis")
    plot_price_distribution(flights)
    plot_destination_frequency(flights)
    plot_price_by_class_and_agency(flights)
    plot_distance_price(flights)
    plot_flight_correlation(flights)
    plot_monthly_travel_demand(flights, hotels)
    plot_monthly_expenditure(flights, hotels)
    plot_price_demand_relationship(flights)
    plot_seasonality(flights, hotels)
    plot_geographical_frequency(flights, hotels)
    adaptive_pricing_exploration(flights)

    banner("SECTION 2 - Hotel & user analysis")
    plot_hotel_place_frequency(hotels)
    plot_hotel_expenditure_distribution(hotels)
    plot_hotel_correlation(hotels)
    plot_user_demographics(users)
    plot_user_expenditure(flights, hotels)
    plot_trip_expenditure(flights, hotels)

    leakage = leakage_analysis(leakage_frame)

    # ---- the same three models, trained two ways --------------------
    random_res, random_preds = train_random_split_regime(ml_frame)
    grouped_res, per_fold, grouped_preds = train_groupkfold_regime(
        ml_frame, n_splits=args.cv_folds)
    comparison = regime_comparison(random_res, grouped_res)

    plot_actual_vs_predicted_regime(
        random_preds, RANDOM_REGIME,
        "ml_actual_vs_predicted_random_split.png",
        f"REGIME {REGIME_A_LABEL} - held-out rows share "
        "configurations with training")
    plot_actual_vs_predicted_regime(
        grouped_preds, GROUPED_REGIME,
        "ml_actual_vs_predicted_groupkfold.png",
        f"REGIME {REGIME_B_LABEL} - whole origin cities withheld")
    plot_metric_bar_comparison(comparison)
    plot_fold_spread(per_fold)

    importance = feature_importance_report(
        ml_frame, perm_sample=args.perm_sample)

    if not args.no_save_model:
        save_deployable_model(ml_frame)

    banner("DONE")
    print(f"  outputs : {OUTPUT_DIR.resolve()}")
    if not args.no_save_model:
        print(f"  models  : {ARTIFACT_DIR.resolve()}")
    print(f"  files   : {len(list(OUTPUT_DIR.glob('*')))} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())