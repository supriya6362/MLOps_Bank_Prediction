from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import f1_score


# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"

MLFLOW_DB = BASE_DIR / "mlflow.db"


# ==================================================
# MLFLOW TRACKING
# ==================================================

mlflow.set_tracking_uri(
    "sqlite:///" + str(MLFLOW_DB).replace("\\", "/")
)


# ==================================================
# LOAD DATA
# ==================================================

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


# ==================================================
# MLFLOW EXPERIMENT
# ==================================================

mlflow.set_experiment(
    "Bank_Marketing_Registry"
)


# ==================================================
# MODEL
# ==================================================

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


# ==================================================
# REGISTERED MODEL NAME
# ==================================================

model_name = "Bank_Marketing_Model"


# ==================================================
# MLFLOW RUN
# ==================================================

with mlflow.start_run(
    run_name="xgb_bank_marketing_registry"
) as run:

    print("\nTraining XGBoost model...")

    model.fit(
        X_train,
        y_train
    )


    # ==================================================
    # PREDICTION
    # ==================================================

    predictions = model.predict(
        X_test
    )


    # ==================================================
    # F1 SCORE
    # ==================================================

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\nF1 Score:", f1)


    # ==================================================
    # LOG PARAMETERS
    # ==================================================

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


    # ==================================================
    # LOG F1 SCORE
    # ==================================================

    mlflow.log_metric(
        "f1_score",
        f1
    )


    # ==================================================
    # LOG XGBOOST MODEL
    # ==================================================

    print("\nLogging model to MLflow...")

    mlflow.xgboost.log_model(
        model,
        name="model",
        registered_model_name=model_name
    )

    print("Model logging completed!")


    # ==================================================
    # SUCCESS MESSAGE
    # ==================================================

    print("\n======================================")
    print(" MODEL REGISTERED SUCCESSFULLY!")
    print("======================================")

    print("Run ID:", run.info.run_id)

    print("Registered Model:", model_name)

    print("F1 Score:", f1)
    print("MLflow Database:", MLFLOW_DB)