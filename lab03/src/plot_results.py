
import csv
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
CSV_PATH = RESULTS_DIR / "training_results.csv"

with open(CSV_PATH, "r", newline="", encoding="utf-8") as file:
    results = list(csv.DictReader(file))

names = [row["model"] for row in results]
accuracies = [float(row["test_accuracy"]) * 100 for row in results]
parameters = [int(row["parameters"]) for row in results]
latencies = [float(row["batch_1_ms"]) for row in results]

plt.figure(figsize=(8, 5))
plt.scatter(parameters, accuracies, s=100)

for name, x, y in zip(names, parameters, accuracies):
    plt.annotate(name, (x, y), xytext=(6, 6),
                 textcoords="offset points")

plt.xlabel("Number of Parameters")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy vs Number of Parameters")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "accuracy_vs_parameters.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(latencies, accuracies, s=100)

for name, x, y in zip(names, latencies, accuracies):
    plt.annotate(name, (x, y), xytext=(6, 6),
                 textcoords="offset points")

plt.xlabel("Inference Latency, Batch 1 (ms)")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy vs Inference Latency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "accuracy_vs_latency.png", dpi=150)
plt.close()

print("Graphs saved successfully!")
print(RESULTS_DIR / "accuracy_vs_parameters.png")
print(RESULTS_DIR / "accuracy_vs_latency.png")
