import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUT = Path("model_outputs")
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# --------------------------------------------------
# 1. Load prepared hotel data
# --------------------------------------------------
train_df = pd.read_csv(OUT / "hotel_train.csv")
test_df = pd.read_csv(OUT / "hotel_test.csv")

print(f"Training samples: {len(train_df)}")
print(f"Testing samples:  {len(test_df)}")

# Target variable
TARGET = "hotel_total"

# --------------------------------------------------
# 2. Define features
# --------------------------------------------------
# Exclude non-feature columns
exclude_cols = [
    "travelCode", "userCode_x", "userCode_y", "origin", "destination", 
    "departure_date", "hotelName", "hotel_price", "hotel_days", "hotel_total",
    "return_origin", "return_destination", "return_date", "flight_rows",
    "hotel_place", "hotel_date", "hotel_userCode", "user_name", "gender", "age", "company",
    "userCode"
]

feature_cols = [c for c in train_df.columns if c not in exclude_cols]
print(f"\nFeature columns ({len(feature_cols)}):")
for c in feature_cols:
    print(f"  {c}")

# Handle duplicate userCode columns
if "userCode_x" in train_df.columns:
    train_df = train_df.drop(columns=["userCode_x", "userCode_y"])
    test_df = test_df.drop(columns=["userCode_x", "userCode_y"])

# Identify categorical and numeric features
categorical_features = train_df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
numeric_features = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

print(f"\nCategorical features ({len(categorical_features)}): {categorical_features}")
print(f"Numeric features ({len(numeric_features)}): {numeric_features}")

# Fill missing values
for df in [train_df, test_df]:
    df[numeric_features] = df[numeric_features].fillna(0)
    df[categorical_features] = df[categorical_features].fillna("unknown")

# --------------------------------------------------
# 3. Define models
# --------------------------------------------------
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ("numeric", StandardScaler(), numeric_features),
])

models = {
    "random_forest": Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ]),
    "gradient_boosting": Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
        )),
    ]),
    "linear_regression": Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]),
}

# --------------------------------------------------
# 4. Train and evaluate
# --------------------------------------------------
X_train = train_df[feature_cols]
y_train = train_df[TARGET]
X_test = test_df[feature_cols]
y_test = test_df[TARGET]

all_metrics = []

for model_name, model in models.items():
    print(f"\n--- Training {model_name} ---")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "model": model_name,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
    
    all_metrics.append(metrics)
    
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    # Save predictions
    pred_df = test_df[["travelCode", "userCode", "destination", "hotelName", "hotel_total"]].copy()
    pred_df["predicted_hotel_total"] = y_pred
    pred_df["residual"] = pred_df["hotel_total"] - pred_df["predicted_hotel_total"]
    pred_df.to_csv(OUT / f"hotel_predictions_{model_name}.csv", index=False)

# --------------------------------------------------
# 5. Save results and select best model
# --------------------------------------------------
metrics_df = pd.DataFrame(all_metrics)
metrics_path = OUT / "hotel_model_comparison.csv"
metrics_df.to_csv(metrics_path, index=False)

# Select best model by R² (primary) then MAE (secondary)
best_idx = metrics_df.sort_values(["r2", "mae"], ascending=[False, True]).index[0]
best_model_info = metrics_df.loc[best_idx]
best_model_name = best_model_info["model"]

print("\n=== BEST HOTEL MODEL ===")
print(f"Model: {best_model_name}")
print(f"R²:    {best_model_info['r2']:.4f}")
print(f"MAE:   {best_model_info['mae']:.4f}")
print(f"RMSE:  {best_model_info['rmse']:.4f}")

# Retrain best model on full training data
best_model = models[best_model_name]
print(f"\nRetraining best model ({best_model_name}) on full training data...")
best_model.fit(X_train, y_train)

# Save best model
model_path = OUT / "best_hotel_model.joblib"
joblib.dump({
    "model": best_model,
    "features": feature_cols,
    "categorical_features": categorical_features,
    "numeric_features": numeric_features,
    "target": TARGET,
    "model_name": best_model_name,
    "metrics": best_model_info.to_dict(),
}, model_path)

print(f"\nBest hotel model saved to: {model_path.resolve()}")
print(f"Metrics saved to: {metrics_path.resolve()}")

# Print summary
print("\n=== FINAL COMPARISON ===")
print(metrics_df[["model", "mae", "rmse", "r2"]]
      .sort_values("r2", ascending=False)
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))