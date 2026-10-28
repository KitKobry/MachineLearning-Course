# Data Card: UCI Adult Income Dataset

## Dataset Source

- **Source:** UCI Adult dataset, accessed through OpenML.
- **Loader:** `fetch_openml(name="adult", version=2, as_frame=True)`.
- **Original dataset size:** 48,842 rows and 15 columns.
- **Input features:** 14.
- **Target:** `class`, representing income category (`<=50K` or `>50K`).

## Intended Use

The dataset is used for educational experiments in data validation, preprocessing, feature engineering, and binary income classification using Logistic Regression.

## Data Processing

- The dataset is split into training and test subsets using a stratified 70/30 split with `random_state=42`.
- Missing numeric values are imputed using the training median.
- Missing categorical values are imputed using the training mode.
- Numeric clipping boundaries are calculated from the training 1st and 99th percentiles.
- Exact duplicate rows are removed from the training subset only.
- The fitted preprocessing components are saved in `results/preprocessor.joblib`.

## Known Limitations and Biases

The dataset contains demographic and employment-related attributes, including age, sex, and race. These variables may reflect historical and societal inequalities. Model predictions should not be interpreted as objective measures of an individual's ability or worth.

The dataset is not necessarily representative of present-day populations or income distributions.

## Missing Data

Missing categorical values occur in `workclass`, `occupation`, and `native-country`. Missing values are imputed during preprocessing.

## Privacy and Ethical Considerations

Although the dataset is publicly available, it contains sensitive socioeconomic and demographic information. Results should be used for educational analysis rather than consequential decisions about individuals.

## License

Refer to the dataset's original distribution page and OpenML metadata for the applicable license and usage terms. The exact license should be verified before redistribution.
