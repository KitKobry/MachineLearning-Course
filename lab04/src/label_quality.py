
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml


LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = LAB_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def audit_label_quality():
    df = fetch_openml(
        name="adult",
        version=2,
        as_frame=True,
    ).frame.copy()

    print("=== LABEL QUALITY AUDIT ===")

    # Audit identical feature rows with conflicting labels.
    feature_cols = [col for col in df.columns if col != "class"]

    grouped = (
        df.groupby(feature_cols, dropna=False, observed=True)["class"]
        .nunique(dropna=False)
    )

    conflict_groups = grouped[grouped > 1]

    print("Conflicting feature groups:", len(conflict_groups))

    # Randomly sample 50 rows for manual review.
    sample = df.sample(n=50, random_state=42).copy()
    sample.insert(0, "row_index", sample.index)

    sample.to_csv(
        RESULTS_DIR / "manual_review_50.csv",
        index=False,
    )

    print("Saved 50 rows for manual review.")

    # Automated checks for the specified range constraints.
    ranges = {
        "age": (17, 90),
        "hours-per-week": (1, 99),
        "capital-gain": (0, 99999),
        "capital-loss": (0, 4356),
    }

    violations = []

    for column, (minimum, maximum) in ranges.items():
        invalid = ~df[column].between(minimum, maximum)

        for idx in df.index[invalid]:
            violations.append({
                "row_index": idx,
                "column": column,
                "value": df.loc[idx, column],
                "expected_range": f"{minimum}-{maximum}",
            })

    pd.DataFrame(violations).to_csv(
        RESULTS_DIR / "label_quality_range_violations.csv",
        index=False,
    )

    summary = pd.DataFrame([
        {
            "check": "conflicting_feature_label_groups",
            "result": len(conflict_groups),
        },
        {
            "check": "manual_review_rows",
            "result": len(sample),
        },
        {
            "check": "range_violations",
            "result": len(violations),
        },
    ])

    summary.to_csv(
        RESULTS_DIR / "label_quality_report.csv",
        index=False,
    )

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print("\nReview manual_review_50.csv against the dataset data dictionary.")


if __name__ == "__main__":
    audit_label_quality()
