from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


# ==================================================
# PROJECT PATH
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MLFLOW_DB = BASE_DIR / "mlflow.db"


# ==================================================
# MLFLOW TRACKING URI
# ==================================================

mlflow.set_tracking_uri(
    "sqlite:///" + str(MLFLOW_DB).replace("\\", "/")
)


# ==================================================
# MLFLOW CLIENT
# ==================================================

client = MlflowClient()


# ==================================================
# REGISTERED MODEL
# ==================================================

MODEL_NAME = "Bank_Marketing_Model"


# ==================================================
# CHECK MODEL
# ==================================================

try:

    registered_model = client.get_registered_model(
        MODEL_NAME
    )

    print("Registered model found!")

    print(
        "Model name:",
        registered_model.name
    )


    # --------------------------------------------------
    # GET MODEL VERSIONS
    # --------------------------------------------------

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )


    print("\nAvailable Model Versions:")

    for version in versions:

        print(
            f"Version: {version.version} | "
            f"Run ID: {version.run_id}"
        )


    # --------------------------------------------------
    # FIND LATEST VERSION
    # --------------------------------------------------

    latest_version = max(
        versions,
        key=lambda x: int(x.version)
    )


    print(
        "\nLatest model version:",
        latest_version.version
    )


    # --------------------------------------------------
    # MODEL URI
    # --------------------------------------------------

    model_uri = (
        f"models:/{MODEL_NAME}/"
        f"{latest_version.version}"
    )


    print(
        "Model URI:",
        model_uri
    )


    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------

    model = mlflow.xgboost.load_model(
        model_uri
    )


    print("\nModel loaded successfully!")


    # --------------------------------------------------
    # SUCCESS
    # --------------------------------------------------

    print("\n======================================")
    print(" MODEL LIFECYCLE CHECK COMPLETED!")
    print("======================================")

    print(
        "Registered Model :",
        MODEL_NAME
    )

    print(
        "Current Version  :",
        latest_version.version
    )

    print(
        "Model Status     : READY FOR USE"
    )


except Exception as e:

    print("\nERROR:")
    print(e)