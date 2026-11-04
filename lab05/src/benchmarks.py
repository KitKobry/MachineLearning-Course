
import random
import time
import statistics

import csv
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import make_moons

from model import MLP, binary_cross_entropy
from value import Value

from pathlib import Path


# Dataset
X, y = make_moons(
    n_samples=200,
    noise=0.1,
    random_state=42
)


# Custom model training
def train_custom():
    random.seed(42)

    model = MLP(2, 8, 1)

    start = time.perf_counter()

    for epoch in range(150):
        for parameter in model.parameters():
            parameter.grad = 0.0

        losses = []

        for features, label in zip(X, y):
            prediction = model(features.tolist())
            loss = binary_cross_entropy(
                prediction, float(label)
            )
            losses.append(loss)

        total_loss = sum(losses, Value(0)) / len(losses)
        total_loss.backward()

        for parameter in model.parameters():
            parameter.data -= 0.05 * parameter.grad

    return time.perf_counter() - start


# PyTorch model training
def train_pytorch():
    torch.manual_seed(42)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(
        y.reshape(-1, 1), dtype=torch.float32
    )

    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    start = time.perf_counter()

    for epoch in range(150):
        optimizer.zero_grad()

        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)

        loss.backward()
        optimizer.step()

    return time.perf_counter() - start


# Benchmark: 5 runs
custom_times = []
pytorch_times = []

for run in range(5):
    custom_times.append(train_custom())
    pytorch_times.append(train_pytorch())

    print(f"Run {run + 1}/5 completed")


# Results
custom_median = statistics.median(custom_times)
pytorch_median = statistics.median(pytorch_times)

print("\nCUSTOM ENGINE TIMES:", custom_times)
print("PYTORCH TIMES:", pytorch_times)

print("\nMedian custom time:", custom_median)
print("Median PyTorch time:", pytorch_median)

print(
    "PyTorch speedup:",
    custom_median / pytorch_median
)

# Save benchmark results
results_dir = Path(__file__).resolve().parents[1] / "results"
results_dir.mkdir(exist_ok=True)

benchmark_data = [
    {
        "run": i + 1,
        "custom_engine_seconds": custom_times[i],
        "pytorch_seconds": pytorch_times[i]
    }
    for i in range(5)
]

with open(results_dir / "benchmark_results.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "run",
            "custom_engine_seconds",
            "pytorch_seconds"
        ]
    )
    writer.writeheader()
    writer.writerows(benchmark_data)

benchmark_summary = {
    "runs": 5,
    "custom_engine_median_seconds": custom_median,
    "pytorch_median_seconds": pytorch_median,
    "pytorch_speedup": custom_median / pytorch_median
}

with open(results_dir / "benchmark_summary.json", "w") as file:
    json.dump(benchmark_summary, file, indent=4)

print("Benchmark results saved to results/")
