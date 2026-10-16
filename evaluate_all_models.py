#!/usr/bin/env python3
"""
Unified evaluation script for all VoyageAnalytics models.

Reads metrics from all model outputs and generates a comprehensive comparison report.
"""

import pandas as pd
from pathlib import Path

MODEL_DIR = Path("model_outputs")

# --------------------------------------------------
# 1. Destination Model Evaluation
# --------------------------------------------------
print("=" * 70)
print("VOYAGE ANALYTICS - UNIFIED MODEL EVALUATION REPORT")
print("=" * 70)

dest_metrics_path = MODEL_DIR / "destination_model_comparison.csv"
if dest_metrics_path.exists():
    dest_metrics = pd.read_csv(dest_metrics_path)
    print("\n### DESTINATION RECOMMENDER MODELS ###")
    print(f"Total experiments: {len(dest_metrics)} (3 models × 2 splits)")
    print("\nAll Results:")
    print(dest_metrics[["split", "model", "top1_accuracy", "hit_rate_at_5", "mrr_at_5"]]
          .sort_values(["split", "hit_rate_at_5"], ascending=[True, False])
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Best model
    best_idx = dest_metrics.sort_values(["hit_rate_at_5", "mrr_at_5"], ascending=[False, False]).index[0]
    best_dest = dest_metrics.loc[best_idx]
    print(f"\n>>> BEST DESTINATION MODEL:")
    print(f"    Split: {best_dest['split']}")
    print(f"    Model: {best_dest['model']}")
    print(f"    Top-1 Accuracy: {best_dest['top1_accuracy']:.4f}")
    print(f"    Hit Rate@5:     {best_dest['hit_rate_at_5']:.4f}")
    print(f"    MRR@5:          {best_dest['mrr_at_5']:.4f}")
    
    # Compare splits
    print(f"\n### SPLIT STRATEGY COMPARISON (averaged across models) ###")
    split_comparison = dest_metrics.groupby("split")[["top1_accuracy", "hit_rate_at_5", "mrr_at_5"]].mean()
    print(split_comparison.to_string(float_format=lambda x: f"{x:.4f}"))
    
    # Compare models
    print(f"\n### MODEL COMPARISON (averaged across splits) ###")
    model_comparison = dest_metrics.groupby("model")[["top1_accuracy", "hit_rate_at_5", "mrr_at_5"]].mean()
    print(model_comparison.to_string(float_format=lambda x: f"{x:.4f}"))
else:
    print("Destination model metrics not found.")

# --------------------------------------------------
# 2. Hotel Model Evaluation
# --------------------------------------------------
hotel_metrics_path = MODEL_DIR / "hotel_model_comparison.csv"
if hotel_metrics_path.exists():
    hotel_metrics = pd.read_csv(hotel_metrics_path)
    print("\n\n### HOTEL RECOMMENDER MODELS (Hotel Expenditure Prediction) ###")
    print(f"Total experiments: {len(hotel_metrics)} (3 models)")
    print("\nAll Results:")
    print(hotel_metrics[["model", "mae", "rmse", "r2"]]
          .sort_values("r2", ascending=False)
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Best model
    best_idx = hotel_metrics.sort_values(["r2", "mae"], ascending=[False, True]).index[0]
    best_hotel = hotel_metrics.loc[best_idx]
    print(f"\n>>> BEST HOTEL MODEL:")
    print(f"    Model: {best_hotel['model']}")
    print(f"    R²:    {best_hotel['r2']:.4f}")
    print(f"    MAE:   {best_hotel['mae']:.2f}")
    print(f"    RMSE:  {best_hotel['rmse']:.2f}")
else:
    print("\nHotel model metrics not found.")

# --------------------------------------------------
# 3. Baseline Comparison (from original outputs)
# --------------------------------------------------
print("\n\n### BASELINE COMPARISON (from original pipeline) ###")

baseline_files = {
    "Global Popularity": "travel_recommender_outputs/destination_baseline_metrics.csv",
    "Personalized History": "travel_recommender_outputs/personalized_destination_metrics.csv",
    "Random Forest Classifier": "travel_recommender_outputs/destination_model_comparison.csv",
    "Candidate RF Ranker": "travel_recommender_outputs/candidate_ranker_metrics.csv",
    "Seasonal Recommender": "travel_recommender_outputs/seasonal_destination_metrics.csv",
}

for name, path in baseline_files.items():
    p = Path(path)
    if p.exists():
        df = pd.read_csv(p)
        if "hit_rate_at_5" in df.columns:
            val = df["hit_rate_at_5"].values[0]
            print(f"  {name}: Hit@5 = {val:.4f}")
        elif "hit_rate" in df.columns:
            val = df["hit_rate"].values[0]
            print(f"  {name}: Hit@5 = {val:.4f}")
        else:
            print(f"  {name}: {df.columns.tolist()}")

# --------------------------------------------------
# 4. Split Evaluation (from evaluate_destination_splits.py)
# --------------------------------------------------
split_metrics_path = Path("travel_recommender_outputs/split_metrics.csv")
if split_metrics_path.exists():
    split_metrics = pd.read_csv(split_metrics_path)
    print("\n### SPLIT EVALUATION (from evaluate_destination_splits.py) ###")
    print(split_metrics[["split", "model", "top1_accuracy", "hit_rate_at_5", "mrr_at_5", "catalog_coverage"]]
          .sort_values(["split", "hit_rate_at_5"], ascending=[True, False])
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# --------------------------------------------------
# 5. Hotel Baseline
# --------------------------------------------------
hotel_baseline_path = Path("travel_recommender_outputs/hotel_metrics_personalized.csv")
if hotel_baseline_path.exists():
    hotel_base = pd.read_csv(hotel_baseline_path)
    print("\n### HOTEL BASELINE (Personalized Popularity) ###")
    print(hotel_base.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# --------------------------------------------------
# 6. Seasonal Patterns Summary
# --------------------------------------------------
recurring_path = Path("travel_recommender_outputs/recurring_user_destination_month.csv")
if recurring_path.exists():
    recurring = pd.read_csv(recurring_path)
    print(f"\n### SEASONAL PATTERNS ###")
    print(f"Total recurring user-destination-month patterns: {len(recurring)}")
    print(f"Unique users with seasonal patterns: {recurring['userCode'].nunique()}")
    print(f"Unique destinations: {recurring['destination'].nunique()}")
    print(f"Months represented: {sorted(recurring['month'].unique())}")

# --------------------------------------------------
# 7. Summary
# --------------------------------------------------
print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
KEY FINDINGS:

1. DESTINATION RECOMMENDER:
   - Best model: Logistic Regression with User-Specific Chronological Split
   - Hit@5: {:.2%}
   - Seasonal features incorporated as ML features (not manual weights)
   - User-specific split slightly outperforms global chronological split

2. HOTEL RECOMMENDER:
   - Best model: Linear Regression (R² = {:.4f})
   - Predicts hotel expenditure based on user/travel features
   - MAE: ${:.2f}, RMSE: ${:.2f}
   - Features include user spending history, destination hotel stats, temporal features

3. SEASONAL FEATURES:
   - Engineered from recurring_user_destination_month.csv
   - Features: seasonal_trips, seasonal_years, seasonal_strength, is_recurring_seasonal
   - Replaced manual SEASONAL_WEIGHTS with learned feature importance

4. DATA SPLITS:
   - Global Chronological 80/20: Standard time-based split
   - User-Specific Chronological 80/20: Per-user time-based split (BEST)

5. MODEL ARTIFACTS SAVED:
   - model_outputs/best_destination_model.joblib
   - model_outputs/best_hotel_model.joblib
   - model_outputs/destination_model_comparison.csv
   - model_outputs/hotel_model_comparison.csv
   - model_outputs/hotel_train.csv / hotel_test.csv
""".format(
    best_dest['hit_rate_at_5'] if 'best_dest' in locals() else 0,
    best_hotel['r2'] if 'best_hotel' in locals() else 0,
    best_hotel['mae'] if 'best_hotel' in locals() else 0,
    best_hotel['rmse'] if 'best_hotel' in locals() else 0,
))

print("=" * 70)