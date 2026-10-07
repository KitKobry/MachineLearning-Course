import os
import time
import statistics
import joblib

from memory_profiler import memory_usage
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42
N_TRAIN_RUNS = 5
N_INFERENCE_RUNS = 100


def create_logistic():
    return LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )


def create_random_forest():
    return RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
    )


def train_model(create_model, X_train, y_train):
    model = create_model()
    model.fit(X_train, y_train)
    return model


def measure_training(create_model, X_train, y_train):
    # Warm-up
    train_model(create_model, X_train, y_train)

    times = []

    for _ in range(N_TRAIN_RUNS):
        start = time.perf_counter()
        train_model(create_model, X_train, y_train)
        end = time.perf_counter()

        times.append(end - start)

    return statistics.median(times)


def measure_inference(model, sample):
    # Warm-up
    model.predict(sample)

    times = []

    for _ in range(N_INFERENCE_RUNS):
        start = time.perf_counter()
        model.predict(sample)
        end = time.perf_counter()

        times.append((end - start) * 1000)

    return statistics.median(times)


def measure_peak_memory(create_model, X_train, y_train):
    memory = memory_usage(
        (train_model, (create_model, X_train, y_train)),
        interval=0.01,
        max_usage=True,
        retval=False,
    )

    return memory


def main():
    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = {
        "LogisticRegression": create_logistic,
        "RandomForest": create_random_forest,
    }

    results = []

    for name, create_model in models.items():
        print(f"\nMeasuring {name}...")

        # Training time
        training_time = measure_training(
            create_model,
            X_train,
            y_train,
        )

        # Final model
        model = train_model(
            create_model,
            X_train,
            y_train,
        )

        # Inference latency
        inference_latency = measure_inference(
            model,
            X_test[0:1],
        )

        # Model size
        model_path = f"lab01/results/{name}.joblib"
        joblib.dump(model, model_path)

        model_size_bytes = os.path.getsize(model_path)
        model_size_kb = model_size_bytes / 1024

        # Peak memory during training
        peak_memory_mb = measure_peak_memory(
            create_model,
            X_train,
            y_train,
        )

        results.append({
            "model": name,
            "training_time_seconds": training_time,
            "inference_latency_ms": inference_latency,
            "model_size_bytes": model_size_bytes,
            "model_size_kb": model_size_kb,
            "peak_memory_mb": peak_memory_mb,
        })

        print(f"Training time: {training_time:.6f} s")
        print(f"Inference latency: {inference_latency:.6f} ms")
        print(f"Model size: {model_size_bytes} bytes")
        print(f"Model size: {model_size_kb:.2f} KB")
        print(f"Peak memory: {peak_memory_mb:.2f} MB")

    output_path = "lab01/results/measurements.csv"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(
            "model,training_time_seconds,"
            "inference_latency_ms,model_size_bytes,"
            "model_size_kb,peak_memory_mb\n"
        )

        for result in results:
            file.write(
                f"{result['model']},"
                f"{result['training_time_seconds']:.6f},"
                f"{result['inference_latency_ms']:.6f},"
                f"{result['model_size_bytes']},"
                f"{result['model_size_kb']:.2f},"
                f"{result['peak_memory_mb']:.2f}\n"
            )

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()