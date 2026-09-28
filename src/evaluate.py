from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models" / "bank_marketing_model.pkl"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

X_test = pd.read_csv(
    DATA_DIR / "X_test.csv"
)

y_test = pd.read_csv(
    DATA_DIR / "y_test.csv"
).values.ravel()


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

predictions = model.predict(X_test)


# --------------------------------------------------
# METRICS
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

print("\nEvaluation Results")
print("-------------------------")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


print("\nConfusion Matrix:")
cm = confusion_matrix(
    y_test,
    predictions
)

print(cm)


# --------------------------------------------------
# SAVE METRICS
# --------------------------------------------------

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

with open(
    OUTPUT_DIR / "metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# --------------------------------------------------
# SAVE EVALUATION CSV
# --------------------------------------------------

results = pd.DataFrame({
    "actual": y_test,
    "predicted": predictions
})

results.to_csv(
    OUTPUT_DIR / "evaluation_results.csv",
    index=False
)


print("\nEvaluation results saved successfully!")