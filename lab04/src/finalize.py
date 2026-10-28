
from pathlib import Path
import hashlib
import json
import yaml
import pandas as pd


LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = LAB_DIR / "results"
CONFIG_PATH = LAB_DIR / "config.yaml"


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    train_path = RESULTS_DIR / "train_processed.csv"
    test_path = RESULTS_DIR / "test_processed.csv"

    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing processed dataset: {path}. Run main.py first."
            )

    # Hash the processed datasets and store reproducibility metadata.
    hashes = {
        "train_processed.csv": sha256_file(train_path),
        "test_processed.csv": sha256_file(test_path),
    }

    hash_path = RESULTS_DIR / "dataset_hashes.json"
    hash_path.write_text(
        json.dumps(hashes, indent=2),
        encoding="utf-8",
    )

    # Verify expected configuration values.
    assert config["dataset"]["expected_rows"] == 48842
    assert config["dataset"]["expected_columns"] == 15
    assert config["split"]["random_state"] == 42

    # Record processed dataset dimensions.
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    summary = {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_columns": len(train.columns),
        "test_columns": len(test.columns),
        "hashes": hashes,
    }

    (RESULTS_DIR / "reproducibility_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=== REPRODUCIBILITY CHECK ===")
    print(json.dumps(summary, indent=2))
    print("Configuration loaded and verified.")
    print("SHA256 hashes saved to results/dataset_hashes.json")


if __name__ == "__main__":
    main()
