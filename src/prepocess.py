from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DATA = BASE_DIR / "data" / "raw" / "bank-additional-full.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading Bank Marketing dataset...")

df = pd.read_csv(RAW_DATA, sep=";")

print("Original dataset shape:", df.shape)


# --------------------------------------------------
# REMOVE DUPLICATES
# --------------------------------------------------

duplicates = df.duplicated().sum()

print("Number of duplicate rows:", duplicates)

df = df.drop_duplicates().reset_index(drop=True)

print("Dataset shape after removing duplicates:", df.shape)


# --------------------------------------------------
# TARGET ENCODING
# --------------------------------------------------

df["y"] = df["y"].map({
    "no": 0,
    "yes": 1
})

print("\nTarget distribution:")
print(df["y"].value_counts())


# --------------------------------------------------
# FEATURES AND TARGET
# --------------------------------------------------

X = df.drop(columns=["y"])
y = df["y"]


# --------------------------------------------------
# IDENTIFY COLUMN TYPES
# --------------------------------------------------

numeric_features = [
    "age",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed"
]

categorical_features = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome"
]


# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# --------------------------------------------------
# PREPROCESSOR
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# --------------------------------------------------
# FIT TRANSFORMER
# --------------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# --------------------------------------------------
# GET FEATURE NAMES
# --------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)


# --------------------------------------------------
# SAVE PROCESSED DATA
# --------------------------------------------------

X_train_processed.to_csv(
    PROCESSED_DIR / "X_train.csv",
    index=False
)

X_test_processed.to_csv(
    PROCESSED_DIR / "X_test.csv",
    index=False
)

y_train.to_csv(
    PROCESSED_DIR / "y_train.csv",
    index=False
)

y_test.to_csv(
    PROCESSED_DIR / "y_test.csv",
    index=False
)


# --------------------------------------------------
# SAVE METADATA
# --------------------------------------------------

metadata = {
    "original_shape": list(df.shape),
    "training_shape": list(X_train_processed.shape),
    "testing_shape": list(X_test_processed.shape),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "target": "y",
    "test_size": 0.20,
    "random_state": 42
}

with open(
    PROCESSED_DIR / "dataset_metadata.json",
    "w"
) as f:
    json.dump(metadata, f, indent=4)


print("\nPreprocessing completed successfully!")
print("Processed data saved inside data/processed/")