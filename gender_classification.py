from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
RANDOM_STATE = 42

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

users = pd.read_csv(BASE / "users.csv")
flights = pd.read_csv(BASE / "flights.csv")
hotels = pd.read_csv(BASE / "hotels.csv")

print("=" * 55)
print("GENDER CLASSIFICATION - MODEL COMPARISON")
print("=" * 55)

# --------------------------------------------------
# 2. VALIDATE REQUIRED COLUMNS
# --------------------------------------------------

required_user_cols = {"code", "gender"}
required_flight_cols = {
    "userCode", "travelCode", "price", "distance",
    "time", "flightType", "agency", "from", "to"
}
required_hotel_cols = {
    "userCode", "travelCode", "total", "days", "place"
}

if not required_user_cols.issubset(users.columns):
    raise ValueError(f"users.csv must contain: {required_user_cols}")

if not required_flight_cols.issubset(flights.columns):
    raise ValueError(f"flights.csv must contain: {required_flight_cols}")

if not required_hotel_cols.issubset(hotels.columns):
    raise ValueError(f"hotels.csv must contain: {required_hotel_cols}")

# Remove missing or non-binary target labels.
users["gender"] = users["gender"].astype("string").str.strip().str.lower()

users = users[
    users["gender"].isin(["male", "female"])
].copy()

# Each user should contribute exactly one target label.
users = users.drop_duplicates(subset="code")

print("\nGender distribution:")
print(users["gender"].value_counts())

# --------------------------------------------------
# 3. AGGREGATE FLIGHT FEATURES BY USER
# --------------------------------------------------

for col in ["price", "distance", "time"]:
    flights[col] = pd.to_numeric(flights[col], errors="coerce")

flight_features = flights.groupby("userCode").agg(
    flight_count=("travelCode", "nunique"),
    avg_flight_price=("price", "mean"),
    total_flight_spend=("price", "sum"),
    avg_distance=("distance", "mean"),
    avg_flight_time=("time", "mean"),
    flight_class_count=("flightType", "nunique"),
    agency_count=("agency", "nunique"),
    origin_count=("from", "nunique"),
    destination_count=("to", "nunique"),
).reset_index()

# --------------------------------------------------
# 4. AGGREGATE HOTEL FEATURES BY USER
# --------------------------------------------------

for col in ["total", "days"]:
    hotels[col] = pd.to_numeric(hotels[col], errors="coerce")

hotel_features = hotels.groupby("userCode").agg(
    hotel_count=("travelCode", "nunique"),
    total_hotel_spend=("total", "sum"),
    avg_hotel_spend=("total", "mean"),
    total_hotel_days=("days", "sum"),
    unique_hotel_places=("place", "nunique"),
).reset_index()

# --------------------------------------------------
# 5. MERGE USER LABELS AND TRAVEL FEATURES
# --------------------------------------------------

data = users[["code", "gender"]].rename(
    columns={"code": "userCode"}
)

data = data.merge(
    flight_features,
    on="userCode",
    how="left",
    validate="one_to_one",
)

data = data.merge(
    hotel_features,
    on="userCode",
    how="left",
    validate="one_to_one",
)

feature_cols = [
    "flight_count",
    "avg_flight_price",
    "total_flight_spend",
    "avg_distance",
    "avg_flight_time",
    "flight_class_count",
    "agency_count",
    "origin_count",
    "destination_count",
    "hotel_count",
    "total_hotel_spend",
    "avg_hotel_spend",
    "total_hotel_days",
    "unique_hotel_places",
]

X = data[feature_cols].replace([np.inf, -np.inf], np.nan)
y = data["gender"]

print("\nTotal labelled users:", len(data))
print("Feature count:", len(feature_cols))
print("Missing feature values:", int(X.isna().sum().sum()))

if y.nunique() != 2 or y.value_counts().min() < 2:
    raise ValueError("Both male and female labels must have at least 2 users.")

# --------------------------------------------------
# 6. HOLD OUT TEST USERS
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

# --------------------------------------------------
# 7. DEFINE MODELS
# --------------------------------------------------

models = {
    "Dummy Baseline": DummyClassifier(strategy="prior"),

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ]),

    "Extra Trees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ]),
}

# --------------------------------------------------
# 8. COMPARE MODELS USING TRAINING DATA ONLY
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

results = []

print("\n5-FOLD TRAINING CROSS-VALIDATION")
print("-" * 55)

for name, model in models.items():
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=1,
    )

    mean_score = scores.mean()
    std_score = scores.std()

    results.append((name, mean_score, std_score))

    print(
        f"{name:22s} "
        f"Balanced accuracy: {mean_score:.3f} "
        f"+/- {std_score:.3f}"
    )

# Choose model using CV results, not test results.
best_name, best_cv_score, _ = max(
    results,
    key=lambda item: item[1],
)

best_model = models[best_name]
best_model.fit(X_train, y_train)

# --------------------------------------------------
# 9. FINAL TEST EVALUATION
# --------------------------------------------------

predictions = best_model.predict(X_test)

print("\nSELECTED MODEL:", best_name)
print("Training CV balanced accuracy:", round(best_cv_score, 4))
print("Test accuracy:", round(accuracy_score(y_test, predictions), 4))
print(
    "Test balanced accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 4),
)
print("Test macro F1:", round(
    f1_score(y_test, predictions, average="macro"), 4
))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=["female", "male"],
    zero_division=0,
))

print("Confusion matrix (female, male):")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["female", "male"],
))

if hasattr(best_model, "predict_proba"):
    probabilities = best_model.predict_proba(X_test)
    classes = list(best_model.classes_)

    if len(classes) == 2:
        positive_index = classes.index("male")
        auc = roc_auc_score(
            (y_test == "male").astype(int),
            probabilities[:, positive_index],
        )
        print("Test ROC-AUC:", round(auc, 4))

# --------------------------------------------------
# 10. SAVE MODEL AND METADATA
# --------------------------------------------------

artifact = {
    "model": best_model,
    "feature_cols": feature_cols,
    "classes": list(best_model.classes_),
    "selected_model": best_name,
    "cross_validation_balanced_accuracy": float(best_cv_score),
    "test_accuracy": float(accuracy_score(y_test, predictions)),
    "test_balanced_accuracy": float(
        balanced_accuracy_score(y_test, predictions)
    ),
}

joblib.dump(artifact, BASE / "gender_model.joblib")

print("\nSaved gender_model.joblib")
print("Evaluation complete.")