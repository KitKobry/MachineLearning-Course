from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = LAB_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def ingest_data():
    """Load the raw Adult dataset without modifying it."""
    dataset = fetch_openml(
        name="adult",
        version=2,
        as_frame=True,
    )

    df = dataset.frame

    print("=== DATA INGESTION ===")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Column names: {df.columns.tolist()}")

    assert df.shape == (48842, 15), (
        f"Unexpected dataset shape: {df.shape}"
    )

    metadata = pd.DataFrame(
        {
            "metric": ["row_count", "column_count"],
            "value": [df.shape[0], df.shape[1]],
        }
    )
    metadata.to_csv(
        RESULTS_DIR / "ingestion_report.csv",
        index=False,
    )

    print("\n=== COLUMN TYPES ===")
    print(df.dtypes)

    print("\n=== FIRST FIVE ROWS ===")
    print(df.head())

    print("\n=== MISSING VALUES ===")
    print(df.isna().sum())

    print("\nIngestion completed successfully.")
    return df


def validate_data(df):
    """Validate schema, ranges, missing values, and labels."""

    print("\n=== SCHEMA VALIDATION ===")

    expected_types = {
        "age": "int",
        "workclass": "category",
        "fnlwgt": "int",
        "education": "category",
        "education-num": "int",
        "marital-status": "category",
        "occupation": "category",
        "relationship": "category",
        "race": "category",
        "sex": "category",
        "capital-gain": "int",
        "capital-loss": "int",
        "hours-per-week": "int",
        "native-country": "category",
        "class": "category",
    }

    schema_violations = []

    for column, expected in expected_types.items():
        actual = str(df[column].dtype)

        if expected == "int" and not pd.api.types.is_integer_dtype(df[column]):
            schema_violations.append(
                f"{column}: expected integer, got {actual}"
            )
        elif expected == "category" and actual != "category":
            schema_violations.append(
                f"{column}: expected category, got {actual}"
            )

    print("Schema violations:", len(schema_violations))
    for violation in schema_violations:
        print(" -", violation)

    print("\n=== RANGE VALIDATION ===")

    ranges = {
        "age": (17, 90),
        "hours-per-week": (1, 99),
        "capital-gain": (0, 99999),
        "capital-loss": (0, 4356),
    }

    range_results = []

    for column, (minimum, maximum) in ranges.items():
        invalid = ~df[column].between(minimum, maximum)
        count = int(invalid.sum())

        range_results.append({
            "column": column,
            "minimum_allowed": minimum,
            "maximum_allowed": maximum,
            "violations": count,
        })

        print(
            f"{column}: allowed [{minimum}, {maximum}], "
            f"violations = {count}"
        )

    range_report = pd.DataFrame(range_results)
    range_report.to_csv(
        RESULTS_DIR / "range_validation.csv",
        index=False,
    )

    print("\n=== MISSING-VALUE VALIDATION ===")

    missing_report = df.isna().sum().rename(
        "missing_count"
    ).rename_axis("column").reset_index()

    print(missing_report.to_string(index=False))

    missing_report.to_csv(
        RESULTS_DIR / "missing_values.csv",
        index=False,
    )

    print("\n=== LABEL VALIDATION ===")

    print("Target column: class")
    print("Label counts:")
    print(df["class"].value_counts(dropna=False))

    print("\nValidation completed.")


def split_data(df):
    """Split raw data before cleaning to avoid data leakage."""

    X = df.drop(columns=["class"])
    y = df["class"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    print("\n=== TRAIN / TEST SPLIT ===")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Train features: {X_train.shape[1]}")
    print(f"Test features: {X_test.shape[1]}")

    print("\nTrain label proportions:")
    print(y_train.value_counts(normalize=True).round(4))

    print("\nTest label proportions:")
    print(y_test.value_counts(normalize=True).round(4))

    split_report = pd.DataFrame({
        "split": ["train", "test"],
        "rows": [len(X_train), len(X_test)],
        "features": [X_train.shape[1], X_test.shape[1]],
    })

    split_report.to_csv(
        RESULTS_DIR / "split_report.csv",
        index=False,
    )

    return X_train, X_test, y_train, y_test



def clean_data(X_train, X_test, y_train, y_test):
    """Fit preprocessing on train only; keep test rows untouched."""

    print("\n=== CLEANING AND TRANSFORMATION ===")

    X_train = X_train.copy()
    X_test = X_test.copy()

    categorical_cols = X_train.select_dtypes(
        include=["category", "object", "string"]
    ).columns.tolist()

    numeric_cols = X_train.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    for column in categorical_cols:
        X_train[column] = X_train[column].astype(object)
        X_test[column] = X_test[column].astype(object)

        X_train[column] = X_train[column].replace("?", np.nan)
        X_test[column] = X_test[column].replace("?", np.nan)

    train_full = X_train.copy()
    train_full["__target__"] = y_train.to_numpy()

    duplicate_mask = train_full.duplicated(keep="first")
    duplicates_removed = int(duplicate_mask.sum())

    X_train = X_train.loc[~duplicate_mask].copy()
    y_train = y_train.loc[~duplicate_mask].copy()

    print("Exact duplicate training rows removed:", duplicates_removed)
    print("Training rows after duplicate removal:", len(X_train))
    print("Test rows (unchanged):", len(X_test))

    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    X_train[numeric_cols] = numeric_imputer.fit_transform(
        X_train[numeric_cols]
    )
    X_test[numeric_cols] = numeric_imputer.transform(
        X_test[numeric_cols]
    )

    X_train[categorical_cols] = categorical_imputer.fit_transform(
        X_train[categorical_cols]
    )
    X_test[categorical_cols] = categorical_imputer.transform(
        X_test[categorical_cols]
    )

    clipping_bounds = {}

    for column in numeric_cols:
        lower = float(X_train[column].quantile(0.01))
        upper = float(X_train[column].quantile(0.99))

        clipping_bounds[column] = (lower, upper)

        X_train[column] = X_train[column].clip(lower, upper)

        X_test[column] = X_test[column].clip(lower, upper)

    train_missing = int(X_train.isna().sum().sum())
    test_missing = int(X_test.isna().sum().sum())

    print("Missing values in processed train:", train_missing)
    print("Missing values in processed test:", test_missing)

    assert train_missing == 0
    assert test_missing == 0

    preprocessor = {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "numeric_imputer": numeric_imputer,
        "categorical_imputer": categorical_imputer,
        "clipping_bounds": clipping_bounds,
    }

    joblib.dump(
        preprocessor,
        RESULTS_DIR / "preprocessor.joblib",
    )

    train_processed = X_train.copy()
    train_processed["class"] = y_train.to_numpy()

    test_processed = X_test.copy()
    test_processed["class"] = y_test.to_numpy()

    train_processed.to_csv(
        RESULTS_DIR / "train_processed.csv",
        index=False,
    )
    test_processed.to_csv(
        RESULTS_DIR / "test_processed.csv",
        index=False,
    )

    pd.DataFrame([
        {
            "metric": "train_rows_after_deduplication",
            "value": len(X_train),
        },
        {
            "metric": "test_rows",
            "value": len(X_test),
        },
        {
            "metric": "train_duplicates_removed",
            "value": duplicates_removed,
        },
        {
            "metric": "train_missing_after_processing",
            "value": train_missing,
        },
        {
            "metric": "test_missing_after_processing",
            "value": test_missing,
        },
    ]).to_csv(
        RESULTS_DIR / "cleaning_report.csv",
        index=False,
    )

    print("Saved preprocessor.joblib")
    print("Saved train_processed.csv and test_processed.csv")

    return X_train, X_test, y_train, y_test


def evaluate_feature_engineering(X_train, X_test, y_train, y_test):
    """Compare Logistic Regression with and without derived features."""

    print("\n=== FEATURE ENGINEERING ===")

    X_train_base = X_train.copy()
    X_test_base = X_test.copy()

    def add_features(X):
        X = X.copy()
        X["capital_net"] = X["capital-gain"] - X["capital-loss"]
        X["hours_per_age"] = (
            X["hours-per-week"] / (X["age"] - 16)
        )
        return X

    X_train_features = add_features(X_train)
    X_test_features = add_features(X_test)

    experiments = {
        "without_derived_features": (X_train_base, X_test_base),
        "with_derived_features": (X_train_features, X_test_features),
    }

    results = []

    for experiment_name, (Xtr, Xte) in experiments.items():
        categorical_cols = Xtr.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        numeric_cols = Xtr.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    StandardScaler(),
                    numeric_cols,
                ),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_cols,
                ),
            ]
        )

        model = Pipeline([
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ])

        model.fit(Xtr, y_train)
        predictions = model.predict(Xte)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(
            y_test,
            predictions,
            pos_label=">50K",
        )

        results.append({
            "experiment": experiment_name,
            "accuracy": accuracy,
            "f1": f1,
            "train_rows": len(Xtr),
            "test_rows": len(Xte),
            "feature_count_before_encoding": Xtr.shape[1],
        })

        print(f"\nExperiment: {experiment_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 (>50K): {f1:.4f}")
        print(f"Input features before encoding: {Xtr.shape[1]}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        RESULTS_DIR / "feature_engineering_results.csv",
        index=False,
    )

    print("\nSaved feature_engineering_results.csv")
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    df = ingest_data()
    validate_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train, X_test, y_train, y_test = clean_data(
        X_train, X_test, y_train, y_test
    )

    evaluate_feature_engineering(
        X_train, X_test, y_train, y_test
    )