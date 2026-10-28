# Lab 4: Data Pipeline and Quality

## 1. Goal

The goal of this laboratory is to build a reproducible data pipeline for the UCI Adult dataset. The pipeline validates the data schema and ranges, handles missing values, removes training duplicates, engineers derived features, evaluates a Logistic Regression baseline, audits label quality, and records dataset provenance.

The target is to predict whether annual income exceeds $50,000. The main evaluation metrics are accuracy and F1-score.

## 2. Method

### 2.1 Data ingestion

The dataset was loaded using `fetch_openml(name="adult", version=2, as_frame=True)`.

- Rows: 48,842
- Columns: 15
- Input features: 14
- Target column in the downloaded dataset: `class`

The target labels are `<=50K` and `>50K`.

### 2.2 Data validation

The validation checks covered expected column types, specified numeric ranges, missing values, and target-label distribution.

| Check | Result |
|---|---:|
| Schema violations | 0 |
| Age outside 17–90 | 0 |
| Hours per week outside 1–99 | 0 |
| Capital gain outside 0–99,999 | 0 |
| Capital loss outside 0–4,356 | 0 |
| Missing `workclass` values | 2,799 |
| Missing `occupation` values | 2,809 |
| Missing `native-country` values | 857 |

The target distribution was 37,155 records with income `<=50K` and 11,687 records with income `>50K`.

### 2.3 Train/test split and cleaning

The dataset was split into training and test sets using a 70/30 stratified split with `random_state=42`, before cleaning.

| Split | Rows before deduplication |
|---|---:|
| Train | 34,189 |
| Test | 14,653 |

Exact duplicate rows were removed from the training data only. A total of 24 training rows were removed, leaving 34,165 training rows. The test set retained all 14,653 rows.

Categorical missing values were imputed using the most frequent value, while numeric missing values were imputed using the median. Numeric clipping boundaries were calculated from the training set's 1st and 99th percentiles and then applied to both sets. No missing values remained after preprocessing.

The fitted imputers and clipping boundaries were saved in `results/preprocessor.joblib`. Processed train and test datasets were saved in the results directory.

### 2.4 Feature engineering and baseline model

Two derived features were added:

- `capital_net = capital-gain - capital-loss`
- `hours_per_age = hours-per-week / (age - 16)`

Logistic Regression was evaluated with and without these derived features. Both experiments used the same training and test split. Numeric features were standardized and categorical features were one-hot encoded within a scikit-learn pipeline fitted on the training data.

### 2.5 Label-quality audit

A reproducible random sample of 50 rows was exported for manual review. No obvious impossible values were identified in the inspected sample.

The automated range audit found zero violations. The duplicate-feature audit identified five groups with conflicting income labels. These groups require documentation because identical recorded features do not necessarily imply that the income labels are correct or incorrect.

The manual review does not independently establish the correctness of the income labels.

## 3. Results

### 3.1 Logistic Regression comparison

| Experiment | Accuracy | F1-score for `>50K` | Input features before encoding |
|---|---:|---:|---:|
| Without derived features | 0.8540 | 0.6601 | 14 |
| With derived features | 0.8578 | 0.6696 | 16 |

Adding the two derived features increased accuracy by approximately 0.0038 and the F1-score by approximately 0.0095. The improvement is modest and has not been tested for statistical significance.

### 3.2 Quality and reproducibility artifacts

The pipeline produced the following artifacts:

- `results/ingestion_report.csv`
- `results/range_validation.csv`
- `results/missing_values.csv`
- `results/split_report.csv`
- `results/cleaning_report.csv`
- `results/preprocessor.joblib`
- `results/train_processed.csv`
- `results/test_processed.csv`
- `results/feature_engineering_results.csv`
- `results/manual_review_50.csv`
- `results/label_quality_range_violations.csv`
- `results/label_quality_report.csv`

## 4. Conclusions

1. The dataset passed the implemented schema and numeric-range checks. Missing categorical values were present and required explicit handling.
2. Removing exact duplicates from training data and fitting imputation and clipping parameters on training data helped prevent data leakage. The test set was not used to fit preprocessing parameters.
3. The derived features improved both accuracy and F1-score in this experiment. The improvement was small, so additional evaluation would be needed before concluding that the features provide a robust benefit.

## 5. Limitations and Reproducibility

The label-quality audit identified five groups of identical feature values with conflicting income labels. A reproducible random sample of 50 rows was manually reviewed, and no obvious impossible values were identified in the inspected sample. This manual review does not independently establish the correctness of the income labels.

The pipeline uses `config.yaml` to document dataset settings, split parameters, preprocessing strategies, feature engineering, model settings, and evaluation metrics. Dataset information and intended use are documented in `data_card.md`.

SHA256 hashes were generated for the processed training and test datasets and saved to `results/dataset_hashes.json`. The reproducibility summary was saved to `results/reproducibility_summary.json`.

The full pipeline can be executed with a single command from the repository root:

```powershell
.\.venv\Scripts\python.exe .\lab04\run_lab04.py
```

The pipeline runs data ingestion, validation, cleaning, feature-engineering experiments, label-quality auditing, and final reproducibility checks.

### Remaining Limitations

- The five conflicting feature groups require further investigation; conflicting labels are not necessarily incorrect.
- The manual review covers only a sample of 50 records.
- The improvement from derived features is modest and has not been tested for statistical significance.
- Reproducibility depends on the dataset version, installed package versions, and execution environment.

