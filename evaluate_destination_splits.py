
"""
Evaluate destination recommenders using:
1. Global chronological 80/20 split
2. User-specific chronological 80/20 split

Models:
- Global popularity baseline
- Personalized history baseline
- Random Forest destination classifier

Input:
    travel_recommender_outputs/journeys.csv

Outputs:
    travel_recommender_outputs/split_metrics.csv
    travel_recommender_outputs/destination_predictions.csv
    travel_recommender_outputs/global_train.csv
    travel_recommender_outputs/global_test.csv
    travel_recommender_outputs/user_train.csv
    travel_recommender_outputs/user_test.csv
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TRAIN_FRACTION = 0.80
TOP_K = 5

DEFAULT_INPUT = Path("travel_recommender_outputs/journeys.csv")
DEFAULT_OUTPUT = Path("travel_recommender_outputs")

CATEGORICAL_FEATURES = ["userCode", "origin"]
NUMERIC_FEATURES = [
    "month_sin",
    "month_cos",
    "day_of_week",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


# ============================================================
# 2. LOAD AND VALIDATE JOURNEYS
# ============================================================

def first_existing_column(df, candidates, description):
    """Return the first available column from a list of aliases."""
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find the {description} column. "
        f"Tried {candidates}. Available columns: {list(df.columns)}"
    )


def load_journeys(input_path: Path) -> pd.DataFrame:
    """Load the journey-level dataset and standardize column names."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path.resolve()}\n"
            "Run the journey preparation script first, or pass --input."
        )

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    date_col = first_existing_column(
        df,
        ["departure_date", "date"],
        "departure date",
    )
    origin_col = first_existing_column(
        df,
        ["origin", "from"],
        "origin",
    )
    destination_col = first_existing_column(
        df,
        ["destination", "to"],
        "destination",
    )

    required = ["userCode"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Standardize the columns used by this script.
    df["departure_date"] = pd.to_datetime(
        df[date_col], errors="coerce"
    )
    df["origin"] = df[origin_col].astype("string").str.strip()
    df["destination"] = (
        df[destination_col].astype("string").str.strip()
    )
    df["userCode"] = df["userCode"].astype("string").str.strip()

    # Remove rows that cannot be used for destination prediction.
    before = len(df)

    df = df.dropna(
        subset=[
            "departure_date",
            "origin",
            "destination",
            "userCode",
        ]
    ).copy()

    invalid_strings = {"", "nan", "None", "<NA>"}

    for col in ["origin", "destination", "userCode"]:
        df = df[~df[col].isin(invalid_strings)]

    df = df.sort_values(
        ["departure_date", "userCode"],
        kind="mergesort",
    ).reset_index(drop=True)

    # A journey-level dataset should have one row per travelCode.
    if "travelCode" in df.columns:
        duplicate_ids = df["travelCode"].duplicated().sum()

        if duplicate_ids:
            raise ValueError(
                f"Found {duplicate_ids} duplicate travelCode values. "
                "Expected one row per journey. Fix journey preparation "
                "before evaluating the models."
            )

    if len(df) < 2:
        raise ValueError("At least two valid journeys are required.")

    if df["destination"].nunique() < 2:
        raise ValueError(
            "At least two distinct destinations are required."
        )

    print("\n=== DATASET ===")
    print(f"Input rows:             {before:,}")
    print(f"Valid journey rows:     {len(df):,}")
    print(f"Users:                  {df['userCode'].nunique():,}")
    print(f"Destinations:           {df['destination'].nunique():,}")
    print(f"Date range:             {df['departure_date'].min()} "
          f"to {df['departure_date'].max()}")

    return df


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features available at prediction time.

    No target-derived historical counts are included in the
    Random Forest. This avoids giving the model different
    definitions of history between the two split strategies.
    """

    out = df.copy()

    month = out["departure_date"].dt.month
    weekday = out["departure_date"].dt.dayofweek

    # Cyclical encoding: December is close to January.
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["day_of_week"] = weekday.astype(int)

    return out


# ============================================================
# 4. SPLIT STRATEGY A: GLOBAL CHRONOLOGICAL 80/20
# ============================================================


def global_chronological_split(df, train_fraction=0.8):
    """
    Split journeys chronologically, targeting the requested training fraction.

    All journeys on the same departure date stay in the same split.
    The cutoff is selected to make the number of training rows as close
    as possible to the target fraction.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    # Count journeys per date, in chronological order.
    counts = (
        df.groupby("departure_date")
        .size()
        .sort_index()
    )

    if len(counts) < 2:
        raise ValueError(
            "At least two distinct departure dates are required "
            "for a chronological train/test split."
        )

    target_train_rows = len(df) * train_fraction

    # Candidate boundaries are between consecutive dates.
    # The final date must remain in the test set.
    cumulative_rows = counts.cumsum().to_numpy()
    possible_train_rows = cumulative_rows[:-1]

    # Choose the boundary that most closely matches the target row count.
    best_boundary_idx = int(
        np.argmin(np.abs(possible_train_rows - target_train_rows))
    )

    # The first test date is immediately after the final training date.
    cutoff = counts.index[best_boundary_idx + 1]

    train = df[df["departure_date"] < cutoff].copy()
    test = df[df["departure_date"] >= cutoff].copy()

    return train, test, cutoff


# ============================================================
# 5. SPLIT STRATEGY B: USER-SPECIFIC CHRONOLOGICAL 80/20
# ============================================================

def user_chronological_split(
    df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
):
    """
    Split each user's journeys chronologically.

    Users with one journey remain in training because they cannot
    contribute both a training and a test journey.
    """

    train_parts = []
    test_parts = []
    single_journey_users = 0

    ordered = df.sort_values(
        ["userCode", "departure_date"],
        kind="mergesort",
    )

    for _, group in ordered.groupby("userCode", sort=False):
        group = group.sort_values(
            "departure_date", kind="mergesort"
        )

        if len(group) < 2:
            train_parts.append(group)
            single_journey_users += 1
            continue

        split_idx = int(len(group) * train_fraction)
        split_idx = min(max(split_idx, 1), len(group) - 1)

        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])

    train = pd.concat(train_parts, ignore_index=True)

    if test_parts:
        test = pd.concat(test_parts, ignore_index=True)
    else:
        test = df.iloc[0:0].copy()

    train = train.sort_values(
        ["departure_date", "userCode"], kind="mergesort"
    ).reset_index(drop=True)

    test = test.sort_values(
        ["departure_date", "userCode"], kind="mergesort"
    ).reset_index(drop=True)

    if train.empty or test.empty:
        raise ValueError(
            "User-specific split produced an empty partition. "
            "Check the number of journeys per user."
        )

    return train, test, single_journey_users


# ============================================================
# 6. MODEL 1: GLOBAL POPULARITY BASELINE
# ============================================================

def rank_global_popularity(train: pd.DataFrame):
    """Rank destinations by their frequency in training data."""

    counts = train["destination"].value_counts()

    return [
        (str(destination), float(count))
        for destination, count in counts.items()
    ]


# ============================================================
# 7. MODEL 2: PERSONALIZED HISTORY BASELINE
# ============================================================

def build_personalized_counts(train: pd.DataFrame):
    """
    Build all personalized baseline counts from training data only.

    These counts are used to rank each test journey's candidate
    destinations. Test labels are never used to construct counts.
    """

    user_route = Counter()
    user_destination = Counter()
    origin_destination = Counter()
    destination_popularity = Counter()

    for row in train[
        ["userCode", "origin", "destination"]
    ].itertuples(index=False, name=None):

        user, origin, destination = map(str, row)

        user_route[(user, origin, destination)] += 1
        user_destination[(user, destination)] += 1
        origin_destination[(origin, destination)] += 1
        destination_popularity[destination] += 1

    return {
        "user_route": user_route,
        "user_destination": user_destination,
        "origin_destination": origin_destination,
        "destination_popularity": destination_popularity,
    }


def rank_personalized(
    user: str,
    origin: str,
    candidates: list[str],
    counts: dict,
):
    """
    Rank candidates lexicographically using:
    1. User's history on this exact route
    2. User's history with the destination
    3. Destination frequency from this origin
    4. Global destination popularity
    """

    user_route = counts["user_route"]
    user_destination = counts["user_destination"]
    origin_destination = counts["origin_destination"]
    destination_popularity = counts["destination_popularity"]

    ranked = []

    for destination in candidates:
        score = (
            user_route[(user, origin, destination)],
            user_destination[(user, destination)],
            origin_destination[(origin, destination)],
            destination_popularity[destination],
        )

        ranked.append((destination, score))

    # Deterministic alphabetical tie-break.
    ranked.sort(key=lambda item: (item[1], item[0]), reverse=True)

    return [
        (destination, float(index))
        for index, (destination, _) in enumerate(ranked)
    ]


# ============================================================
# 8. MODEL 3: RANDOM FOREST DESTINATION CLASSIFIER
# ============================================================

def make_random_forest():
    """Create the categorical + calendar-feature classifier."""

    preprocess = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", classifier),
        ]
    )


def rank_random_forest(model, row: pd.DataFrame):
    """Return destinations ranked by the classifier's scores."""

    probabilities = model.predict_proba(row[MODEL_FEATURES])[0]
    classes = model.named_steps["model"].classes_

    order = np.argsort(probabilities)[::-1]

    return [
        (str(classes[i]), float(probabilities[i]))
        for i in order[:TOP_K]
    ]


# ============================================================
# 9. METRICS
# ============================================================

def calculate_metrics(
    split_name: str,
    model_name: str,
    predictions: pd.DataFrame,
    full_catalog: set[str],
):
    """Calculate ranking metrics from per-journey predictions."""

    if predictions.empty:
        return {
            "split": split_name,
            "model": model_name,
            "test_journeys": 0,
            "test_users": 0,
            "top1_accuracy": np.nan,
            "hit_rate_at_5": np.nan,
            "mrr_at_5": np.nan,
            "catalog_coverage": np.nan,
        }

    top1 = []
    hit5 = []
    reciprocal_ranks = []
    recommended_catalog = set()

    for row in predictions.itertuples(index=False):
        actual = str(row.actual_destination)
        recommended = row.recommended_destinations

        recommended_catalog.update(recommended)

        top1.append(
            int(bool(recommended) and recommended[0] == actual)
        )

        hit5.append(int(actual in recommended[:TOP_K]))

        if actual in recommended[:TOP_K]:
            rank = recommended.index(actual) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    coverage = (
        len(recommended_catalog) / len(full_catalog)
        if full_catalog else 0.0
    )

    return {
        "split": split_name,
        "model": model_name,
        "test_journeys": len(predictions),
        "test_users": predictions["userCode"].nunique(),
        "top1_accuracy": float(np.mean(top1)),
        "hit_rate_at_5": float(np.mean(hit5)),
        "mrr_at_5": float(np.mean(reciprocal_ranks)),
        "catalog_coverage": float(coverage),
    }


# ============================================================
# 10. EVALUATE ALL MODELS ON ONE SPLIT
# ============================================================

def evaluate_split(
    split_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    full_catalog: set[str],
):
    """
    Fit models on this split's training data and evaluate on its
    held-out test data.
    """

    print(f"\n{'=' * 65}")
    print(f"Evaluating split: {split_name}")
    print(f"{'=' * 65}")

    print(f"Training journeys: {len(train):,}")
    print(f"Test journeys:     {len(test):,}")
    print(f"Training users:    {train['userCode'].nunique():,}")
    print(f"Test users:        {test['userCode'].nunique():,}")
    print(f"Train destinations:{train['destination'].nunique():>5,}")
    print(f"Test destinations: {test['destination'].nunique():,}")

    if train.empty or test.empty:
        raise ValueError(f"{split_name} has an empty partition.")

    # A destination absent from training cannot be predicted by
    # any of these supervised approaches.
    train_catalog = sorted(train["destination"].astype(str).unique())

    # Build the training-only popularity ranking.
    popularity_ranking = rank_global_popularity(train)

    # Build training-only personalized counts.
    personalized_counts = build_personalized_counts(train)

    # Fit Random Forest.
    model = make_random_forest()

    train_features = add_calendar_features(train)
    test_features = add_calendar_features(test)

    model.fit(
        train_features[MODEL_FEATURES],
        train_features["destination"].astype(str),
    )

    # Use consistent candidate destinations for the two baselines.
    # Random Forest can rank only classes seen during training.
    rows_by_model = defaultdict(list)

    for idx, row in test_features.iterrows():
        user = str(row["userCode"])
        origin = str(row["origin"])
        actual = str(row["destination"])

        # Global popularity.
        global_top = [
            destination
            for destination, _ in popularity_ranking[:TOP_K]
        ]

        # Personalized ranking.
        personalized_ranked = rank_personalized(
            user=user,
            origin=origin,
            candidates=train_catalog,
            counts=personalized_counts,
        )

        personalized_top = [
            destination
            for destination, _ in personalized_ranked[:TOP_K]
        ]

        # Random Forest ranking.
        rf_ranked = rank_random_forest(
            model,
            test_features.loc[[idx]],
        )

        rf_top = [destination for destination, _ in rf_ranked]

        recommendations = {
            "global_popularity": global_top,
            "personalized_baseline": personalized_top,
            "random_forest": rf_top,
        }

        for model_name, ranked_destinations in recommendations.items():
            rows_by_model[model_name].append({
                "split": split_name,
                "model": model_name,
                "userCode": user,
                "departure_date": row["departure_date"],
                "origin": origin,
                "actual_destination": actual,
                "recommended_destinations": ranked_destinations,
                "top1_correct": (
                    bool(ranked_destinations)
                    and ranked_destinations[0] == actual
                ),
                "hit_at_5": int(actual in ranked_destinations[:TOP_K]),
                "reciprocal_rank_at_5": (
                    1.0 / (ranked_destinations.index(actual) + 1)
                    if actual in ranked_destinations[:TOP_K]
                    else 0.0
                ),
                "actual_destination_seen_in_training": (
                    actual in train_catalog
                ),
            })

    metrics = []
    prediction_frames = []

    for model_name, rows in rows_by_model.items():
        pred_df = pd.DataFrame(rows)

        metrics.append(
            calculate_metrics(
                split_name=split_name,
                model_name=model_name,
                predictions=pred_df,
                full_catalog=full_catalog,
            )
        )

        # Serialize list columns for CSV.
        pred_df["recommended_destinations"] = pred_df[
            "recommended_destinations"
        ].apply(lambda values: "|".join(values))

        prediction_frames.append(pred_df)

    return metrics, prediction_frames


# ============================================================
# 11. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare destination recommenders using global and "
            "user-specific chronological 80/20 splits."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to journeys.csv",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for metrics, predictions and split CSVs",
    )

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    journeys = load_journeys(args.input)
    full_catalog = set(journeys["destination"].astype(str).unique())

    # Add calendar features once; these features do not depend on labels.
    journeys = add_calendar_features(journeys)

    # --------------------------------------------------------
    # Create both splits.
    # --------------------------------------------------------

    global_train, global_test, cutoff = global_chronological_split(
        journeys
    )

    user_train, user_test, single_journey_users = (
        user_chronological_split(journeys)
    )

    print("\n=== SPLIT SUMMARY ===")
    print(f"Global chronological cutoff: {cutoff}")
    print(
        f"Global split: train={len(global_train):,}, "
        f"test={len(global_test):,}"
    )
    print(
        f"User split:   train={len(user_train):,}, "
        f"test={len(user_test):,}"
    )
    print(
        "Users with only one journey retained in training: "
        f"{single_journey_users:,}"
    )

    # Save the split datasets for reproducibility.
    global_train.to_csv(
        args.output / "global_train.csv", index=False
    )
    global_test.to_csv(
        args.output / "global_test.csv", index=False
    )
    user_train.to_csv(
        args.output / "user_train.csv", index=False
    )
    user_test.to_csv(
        args.output / "user_test.csv", index=False
    )

    # --------------------------------------------------------
    # Evaluate all models on both splits.
    # --------------------------------------------------------

    all_metrics = []
    all_predictions = []

    experiments = [
        ("global_80_20", global_train, global_test),
        ("user_80_20", user_train, user_test),
    ]

    for split_name, train, test in experiments:
        metrics, prediction_frames = evaluate_split(
            split_name=split_name,
            train=train,
            test=test,
            full_catalog=full_catalog,
        )

        all_metrics.extend(metrics)
        all_predictions.extend(prediction_frames)

    # --------------------------------------------------------
    # Save outputs.
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(all_metrics)

    metrics_path = args.output / "split_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    if all_predictions:
        predictions_df = pd.concat(
            all_predictions,
            ignore_index=True,
        )

        predictions_path = (
            args.output / "destination_predictions.csv"
        )

        predictions_df.to_csv(
            predictions_path,
            index=False,
        )
    else:
        predictions_path = None

    print("\n=== FINAL COMPARISON ===")

    display_columns = [
        "split",
        "model",
        "test_journeys",
        "top1_accuracy",
        "hit_rate_at_5",
        "mrr_at_5",
        "catalog_coverage",
    ]

    print(
        metrics_df[display_columns]
        .sort_values(["split", "hit_rate_at_5"], ascending=[True, False])
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nSaved outputs:")
    print(f"Metrics:     {metrics_path.resolve()}")

    if predictions_path:
        print(f"Predictions: {predictions_path.resolve()}")

    print(f"Output dir:  {args.output.resolve()}")


if __name__ == "__main__":
    main()
