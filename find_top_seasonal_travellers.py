#!/usr/bin/env python3
"""
Find the top 5 users with recurring seasonal destination patterns.

Input:
  travel_recommender_outputs/journeys_train.csv
  users.csv

Outputs:
  seasonal_travellers/top_5_seasonal_travellers.txt
  seasonal_travellers/top_5_seasonal_travellers.png
  seasonal_travellers/seasonal_traveller_patterns.csv

A recurring seasonal pattern is defined as:
  - same user + same destination + same departure month
  - at least 2 trips in that month
  - across at least 2 distinct years

Ranking:
  Users are ranked by the total number of trips belonging to qualifying
  recurring user-destination-month patterns. Users with multiple seasonal
  destinations have each pattern listed separately.
"""

from pathlib import Path
import argparse
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_JOURNEYS = Path("travel_recommender_outputs/journeys_train.csv")
DEFAULT_USERS = Path("users.csv")
DEFAULT_OUTPUT = Path("seasonal_travellers")


def load_data(journeys_path: Path, users_path: Path):
    journeys = pd.read_csv(journeys_path)
    users = pd.read_csv(users_path)

    required_journey_cols = {"userCode", "origin", "destination", "departure_date"}
    required_user_cols = {"code", "name"}
    missing_journeys = required_journey_cols - set(journeys.columns)
    missing_users = required_user_cols - set(users.columns)

    if missing_journeys:
        raise ValueError(f"Missing columns in {journeys_path}: {sorted(missing_journeys)}")
    if missing_users:
        raise ValueError(f"Missing columns in {users_path}: {sorted(missing_users)}")

    journeys["userCode"] = journeys["userCode"].astype(str).str.strip()
    users["code"] = users["code"].astype(str).str.strip()
    journeys["departure_date"] = pd.to_datetime(
        journeys["departure_date"], errors="coerce"
    )
    journeys = journeys.dropna(
        subset=["departure_date", "userCode", "origin", "destination"]
    ).copy()

    for col in ["origin", "destination"]:
        journeys[col] = journeys[col].astype(str).str.strip()

    users["name"] = users["name"].fillna("Unknown").astype(str).str.strip()
    # Keep one row per user code to prevent duplicate joins.
    users = users.drop_duplicates(subset=["code"], keep="first")
    return journeys, users


def find_seasonal_patterns(journeys: pd.DataFrame, users: pd.DataFrame):
    data = journeys.copy()
    data["year"] = data["departure_date"].dt.year
    data["month"] = data["departure_date"].dt.month
    data["month_name"] = data["departure_date"].dt.strftime("%B")

    # One row per qualifying user + origin + destination + month pattern.
    # Origin is included so that each route is shown separately.
    pattern_stats = (
        data.groupby(
            ["userCode", "origin", "destination", "month", "month_name"],
            dropna=False,
        )
        .agg(
            total_frequency=("departure_date", "size"),
            distinct_years=("year", "nunique"),
            first_date=("departure_date", "min"),
            last_date=("departure_date", "max"),
        )
        .reset_index()
    )

    recurring = pattern_stats[
        (pattern_stats["total_frequency"] >= 2)
        & (pattern_stats["distinct_years"] >= 2)
    ].copy()

    # Dates are all observed journeys in each qualifying pattern, sorted chronologically.
    key_cols = ["userCode", "origin", "destination", "month"]
    qualifying_keys = recurring[key_cols].drop_duplicates()
    dated = data.merge(qualifying_keys, on=key_cols, how="inner")
    dates = (
        dated.sort_values("departure_date")
        .groupby(key_cols)["departure_date"]
        .apply(lambda s: ", ".join(pd.to_datetime(s).dt.strftime("%Y-%m-%d")))
        .rename("travel_dates")
        .reset_index()
    )
    recurring = recurring.merge(dates, on=key_cols, how="left")

    # Join display name.
    recurring = recurring.merge(
        users[["code", "name"]].rename(columns={"code": "userCode", "name": "name"}),
        on="userCode",
        how="left",
    )
    recurring["name"] = recurring["name"].fillna("Unknown")

    # Score each user by all trips across their qualifying seasonal patterns.
    # The total includes each pattern's trip count; separate routes/months are listed.
    user_scores = (
        recurring.groupby(["userCode", "name"], dropna=False)
        .agg(
            total_frequency=("total_frequency", "sum"),
            seasonal_patterns=("destination", "size"),
            seasonal_destinations=("destination", "nunique"),
            seasonal_months=("month", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["total_frequency", "seasonal_patterns", "userCode"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    recurring = recurring.merge(
        user_scores[["userCode", "total_frequency"]].rename(
            columns={"total_frequency": "user_total_frequency"}
        ),
        on="userCode",
        how="left",
    )
    recurring = recurring.sort_values(
        ["user_total_frequency", "userCode", "month", "destination", "origin"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)

    top_codes = user_scores.head(5)["userCode"].tolist()
    top_patterns = recurring[recurring["userCode"].isin(top_codes)].copy()
    # Preserve user ranking in report.
    rank_map = {code: rank + 1 for rank, code in enumerate(top_codes)}
    top_patterns["rank"] = top_patterns["userCode"].map(rank_map)
    top_patterns = top_patterns.sort_values(
        ["rank", "month", "destination", "origin"]
    )
    return user_scores, top_patterns


def write_text_report(user_scores: pd.DataFrame, top_patterns: pd.DataFrame, output_path: Path):
    lines = [
        "TOP 5 SEASONAL TRAVELLERS",
        "=" * 72,
        "Pattern definition: same user, origin, destination and departure month;",
        "at least 2 trips across at least 2 distinct years.",
        "Ranking frequency = sum of trips in qualifying seasonal route-month patterns.",
        "",
    ]

    for rank, (_, user) in enumerate(user_scores.head(5).iterrows(), start=1):
        code = str(user["userCode"])
        name = str(user["name"])
        lines.extend([
            "-" * 72,
            f"Rank: {rank}",
            f"User code: {code}    Name: {name}",
            "",
        ])

        user_rows = top_patterns[top_patterns["userCode"] == code].sort_values(
            ["month", "destination", "origin"]
        )
        if user_rows.empty:
            lines.append("No qualifying seasonal patterns found.")
            continue

        for _, row in user_rows.iterrows():
            lines.extend([
                f"Origin: {row['origin']}",
                f"Destination: {row['destination']}",
                f"Periodic month: {row['month_name']}",
                f"Dates travelled: {row['travel_dates']}",
                f"Pattern frequency: {int(row['total_frequency'])}",
                f"Years represented: {int(row['distinct_years'])}",
                "-" * 40,
            ])

        lines.extend([
            f"Total frequency: {int(user['total_frequency'])}",
            f"Qualifying patterns: {int(user['seasonal_patterns'])}",
            f"Distinct seasonal destinations: {int(user['seasonal_destinations'])}",
            "",
        ])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(user_scores: pd.DataFrame, top_patterns: pd.DataFrame, output_path: Path):
    top_users = user_scores.head(5).copy()
    top_users["label"] = top_users.apply(
        lambda r: f"{r['name']} (code {r['userCode']})", axis=1
    )
    top_users = top_users.sort_values("total_frequency", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(top_users["label"], top_users["total_frequency"])
    ax.bar_label(bars, padding=4, fmt="%.0f")
    ax.set_title("Top 5 Seasonal Travellers")
    ax.set_xlabel("Trips in recurring seasonal route-month patterns")
    ax.set_ylabel("User")
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journeys", type=Path, default=DEFAULT_JOURNEYS)
    parser.add_argument("--users", type=Path, default=DEFAULT_USERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    journeys, users = load_data(args.journeys, args.users)
    user_scores, top_patterns = find_seasonal_patterns(journeys, users)

    # Save the detailed seasonal patterns for every user, not just the top five.
    all_scores, all_patterns = find_seasonal_patterns(journeys, users)
    all_patterns.to_csv(args.output_dir / "seasonal_traveller_patterns.csv", index=False)

    write_text_report(
        user_scores,
        top_patterns,
        args.output_dir / "top_5_seasonal_travellers.txt",
    )
    write_plot(
        user_scores,
        top_patterns,
        args.output_dir / "top_5_seasonal_travellers.png",
    )

    print("TOP 5 SEASONAL TRAVELLERS")
    print("=" * 72)
    for rank, (_, user) in enumerate(user_scores.head(5).iterrows(), start=1):
        print(
            f"{rank}. User code: {user['userCode']} | Name: {user['name']} | "
            f"Total frequency: {int(user['total_frequency'])} | "
            f"Patterns: {int(user['seasonal_patterns'])} | "
            f"Destinations: {int(user['seasonal_destinations'])}"
        )
        rows = top_patterns[top_patterns["userCode"] == user["userCode"]]
        for _, row in rows.iterrows():
            print(
                f"   {row['origin']} -> {row['destination']} | "
                f"{row['month_name']} | Dates: {row['travel_dates']} | "
                f"Frequency: {int(row['total_frequency'])}"
            )

    print("\nFiles written:")
    print(f"- {args.output_dir / 'top_5_seasonal_travellers.txt'}")
    print(f"- {args.output_dir / 'top_5_seasonal_travellers.png'}")
    print(f"- {args.output_dir / 'seasonal_traveller_patterns.csv'}")


if __name__ == "__main__":
    main()
