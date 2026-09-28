from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"

MLFLOW_DB = BASE_DIR / "mlflow.db"


# --------------------------------------------------
# MLflow Tracking Database
# --------------------------------------------------

mlflow.set_tracking_uri(
    "sqlite:///" + str(MLFLOW_DB).replace("\\", "/")
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

X_train = pd.read_csv(
    DATA_DIR / "X_train.csv"
)

X_test = pd.read_csv(
    DATA_DIR / "X_test.csv"
)

y_train = pd.read_csv(
    DATA_DIR / "y_train.csv"
).values.ravel()

y_test = pd.read_csv(
    DATA_DIR / "y_test.csv"
).values.ravel()


print("Data loaded successfully!")

print("Training data:", X_train.shape)

print("Testing data:", X_test.shape)


# --------------------------------------------------
# MLflow Experiment
# --------------------------------------------------

mlflow.set_experiment(
    "Bank_Marketing_Prediction"
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)


# --------------------------------------------------
# MLflow RUN
# --------------------------------------------------

with mlflow.start_run(
    run_name="xgb_bank_marketing"
):

    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )


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
    # LOG PARAMETERS
    # --------------------------------------------------

    mlflow.log_param(
        "n_estimators",
        300
    )

    mlflow.log_param(
        "learning_rate",
        0.03
    )

    mlflow.log_param(
        "max_depth",
        3
    )

    mlflow.log_param(
        "min_child_weight",
        2
    )

    mlflow.log_param(
        "subsample",
        0.8
    )

    mlflow.log_param(
        "colsample_bytree",
        0.8
    )


    # --------------------------------------------------
    # LOG METRICS
    # --------------------------------------------------

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )


    # --------------------------------------------------
    # LOG MODEL
    # --------------------------------------------------

    mlflow.sklearn.log_model(
        model,
        "model"
    )


    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    print("\n======================================")
    print(" MLflow Training Completed!")
    print("======================================")

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print(
        "\nRun ID:",
        mlflow.active_run().info.run_id
    )

    print(
        "Experiment:",
        "Bank_Marketing_Prediction"
    )

    mlflow.sklearn.log_model(
    model,
    name="model",
    skops_trusted_types=[
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBClassifier"
    ]
)