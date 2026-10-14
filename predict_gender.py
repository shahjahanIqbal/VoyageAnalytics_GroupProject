from pathlib import Path
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent

artifact = joblib.load(BASE / "gender_model.joblib")
model = artifact["model"]
feature_cols = artifact["feature_cols"]


def predict_label(travel_features):
    """
    travel_features: dictionary containing the model's
    required numerical travel features.

    Returns the predicted dataset label.
    """
    row = pd.DataFrame([travel_features])
    row = row.reindex(columns=feature_cols)

    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]

    return {
        "predicted_label": prediction,
        "class_probabilities": dict(
            zip(artifact["classes"], probabilities.tolist())
        ),
        "model_test_accuracy": artifact["test_accuracy"],
        "warning": (
            "Experimental model with weak test performance. "
            "Do not rely on this prediction to determine "
            "an individual's gender."
        ),
    }