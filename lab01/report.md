# Lab 1: Environment and First System Measurements

## Goal

The goal of this laboratory was to create a reproducible machine learning environment, train two baseline classification models, measure their system costs, and evaluate their suitability for different deployment environments.

The two baseline models were Logistic Regression and Random Forest, trained on the Breast Cancer Wisconsin (Diagnostic) dataset.

## Method

The laboratory was performed using Python 3.11.8 with the required pinned dependencies.

The Breast Cancer Wisconsin dataset was loaded using `sklearn.datasets.load_breast_cancer`.

The dataset was divided into training and testing sets using a 70/30 stratified split with `random_state=42`.

Two baseline models were trained:

- Logistic Regression with `max_iter=1000` and `random_state=42`.
- Random Forest with `n_estimators=100` and `random_state=42`.

Model accuracy was measured on the test set.

System measurements were performed for both models:

- Training time was measured using one warm-up run followed by five measured runs, with the median reported.
- Single-sample inference latency was measured using one warm-up followed by 100 measured runs, with the median reported.
- Model size was measured after saving each model with `joblib`.
- Peak memory usage during training was measured using `memory_profiler`.

All experiments used `random_state=42` for reproducibility.

## Results

### Classification Accuracy

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 0.9474 |
| Random Forest | 0.9357 |

### System Measurements

| Model | Training Time (s) | Inference Latency (ms) | Model Size (KB) | Peak Memory (MB) |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.800949 | 0.191500 | 1.03 | 106.32 |
| Random Forest | 0.459769 | 10.081500 | 284.07 | 107.57 |

### Deployment Suitability

The laboratory specifies the following deployment budgets:

| Platform | Memory | Latency | Model Size |
|---|---:|---:|---:|
| Cloud | ≥ 1 GB | ≤ 100 ms | ≤ 500 MB |
| Edge | 256–1024 MB | ≤ 50 ms | ≤ 50 MB |
| Mobile | 64–256 MB | ≤ 20 ms | ≤ 10 MB |
| TinyML | ≤ 256 KB | ≤ 10 ms | ≤ 100 KB |

Based on the measured values:

| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| Logistic Regression | Suitable | Suitable | Suitable | Not suitable |
| Random Forest | Suitable | Suitable | Suitable | Not suitable |

Both models satisfy the measured latency and model-size requirements for Cloud, Edge, and Mobile deployment.

Neither model is suitable for TinyML because their measured peak memory usage is far above the TinyML memory budget of 256 KB. Random Forest also exceeds the TinyML model-size limit of 100 KB.

## Conclusions

1. Logistic Regression achieved the higher test accuracy of 0.9474 compared with 0.9357 for Random Forest. It also had substantially lower inference latency and a much smaller model size.

2. Random Forest had lower measured training time than Logistic Regression (0.459769 s compared with 0.800949 s), but its inference latency and model size were significantly higher.

3. Based on the measured resource requirements and the specified budgets, both models are suitable for Cloud, Edge, and Mobile deployment, but neither is suitable for TinyML because of the memory requirements.