import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


digits = load_digits()

X = digits.data.astype(np.float64) / 16.0

y = digits.target

num_classes = 10
Y = np.eye(num_classes)[y]

X_train, X_test, Y_train, Y_test, y_train, y_test = train_test_split(
    X,
    Y,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("Dataset loaded successfully!")
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)
print("Input range:", X.min(), "-", X.max())
print("First label:", y[0])
print("First one-hot label:", Y[0])


# Task 2

def initialize_parameters(width, seed=42):
    rng = np.random.default_rng(seed)

    W1 = rng.normal(0, 0.01, size=(64, width))
    b1 = np.zeros(width)

    W2 = rng.normal(0, 0.01, size=(width, width))
    b2 = np.zeros(width)

    W3 = rng.normal(0, 0.01, size=(width, 10))
    b3 = np.zeros(10)

    return W1, b1, W2, b2, W3, b3

for width in [16, 64]:
    W1, b1, W2, b2, W3, b3 = initialize_parameters(width)

    print(f"\nNetwork width: {width}")
    print("W1 shape:", W1.shape)
    print("b1 shape:", b1.shape)
    print("W2 shape:", W2.shape)
    print("b2 shape:", b2.shape)
    print("W3 shape:", W3.shape)
    print("b3 shape:", b3.shape)


# Task 3

def relu(z):
    return np.maximum(0, z)


def softmax(z):
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward_propagation(X, parameters):
    W1, b1, W2, b2, W3, b3 = parameters

    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    Z3 = A2 @ W3 + b3
    probabilities = softmax(Z3)

    cache = (X, Z1, A1, Z2, A2, Z3, probabilities)

    return probabilities, cache


for width in [16, 64]:
    parameters = initialize_parameters(width)

    probabilities, cache = forward_propagation(
        X_train[:32], parameters
    )

    print(f"\nForward propagation, width={width}")
    print("Output shape:", probabilities.shape)
    print("First sample probabilities:", probabilities[0])
    print("Sum of probabilities:", probabilities[0].sum())
    print(
        "All rows sum to 1:",
        np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    )

    
# Task 4

def cross_entropy_loss(Y_true, probabilities):
    eps = 1e-12
    probabilities = np.clip(probabilities, eps, 1.0)

    loss = -np.sum(Y_true * np.log(probabilities)) / Y_true.shape[0]
    return loss


print("\nTask 4: Cross-entropy loss")

for width in [16, 64]:
    parameters = initialize_parameters(width)

    probabilities, _ = forward_propagation(
        X_train, parameters
    )

    loss = cross_entropy_loss(Y_train, probabilities)

    print(f"Width {width}: loss = {loss:.6f}")

print("Expected loss for uniform predictions:", np.log(10))


# Task 5

def backward_propagation(Y_true, parameters, cache):
    W1, b1, W2, b2, W3, b3 = parameters
    X, Z1, A1, Z2, A2, Z3, probabilities = cache

    m = X.shape[0]

    dZ3 = (probabilities - Y_true) / m
    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0)

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * (Z2 > 0)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    gradients = (dW1, db1, dW2, db2, dW3, db3)
    return gradients


print("\nTask 5: Backpropagation")

parameters = initialize_parameters(16)
probabilities, cache = forward_propagation(X_train[:32], parameters)
gradients = backward_propagation(Y_train[:32], parameters, cache)

parameter_names = ["dW1", "db1", "dW2", "db2", "dW3", "db3"]

for name, gradient in zip(parameter_names, gradients):
    print(f"{name} shape: {gradient.shape}")
    print(f"{name} all finite: {np.all(np.isfinite(gradient))}")


# Numerical gradient checking


def gradient_check(X, Y, parameters, num_checks=10, epsilon=1e-5):
    rng = np.random.default_rng(123)

    weight_indices = [0, 2, 4]
    passed = 0
    checks_done = 0
    attempts = 0
    max_attempts = 1000

    while checks_done < num_checks and attempts < max_attempts:
        attempts += 1

        param_index = int(rng.choice(weight_indices))
        param = parameters[param_index]

        index = tuple(
            int(rng.integers(0, size))
            for size in param.shape
        )

        original_value = param[index]

        _, baseline_cache = forward_propagation(X, parameters)
        baseline_mask1 = baseline_cache[1] > 0
        baseline_mask2 = baseline_cache[3] > 0

        param[index] = original_value + epsilon
        p_plus, cache_plus = forward_propagation(X, parameters)
        loss_plus = cross_entropy_loss(Y, p_plus)

        param[index] = original_value - epsilon
        p_minus, cache_minus = forward_propagation(X, parameters)
        loss_minus = cross_entropy_loss(Y, p_minus)

        param[index] = original_value

        stable_relu = (
            np.array_equal(baseline_mask1, cache_plus[1] > 0)
            and np.array_equal(baseline_mask1, cache_minus[1] > 0)
            and np.array_equal(baseline_mask2, cache_plus[3] > 0)
            and np.array_equal(baseline_mask2, cache_minus[3] > 0)
        )

        if not stable_relu:
            continue

        numerical = (loss_plus - loss_minus) / (2 * epsilon)

        _, cache = forward_propagation(X, parameters)
        gradients = backward_propagation(Y, parameters, cache)
        analytic = gradients[param_index][index]

        denominator = abs(analytic) + abs(numerical)

        if denominator < 1e-12:
            relative_error = 0.0
        else:
            relative_error = abs(analytic - numerical) / denominator

        success = relative_error < 1e-6
        passed += int(success)
        checks_done += 1

        print(
            f"Check {checks_done}: "
            f"{parameter_names[param_index][1:]}{index}, "
            f"analytic={analytic:.10e}, "
            f"numerical={numerical:.10e}, "
            f"relative error={relative_error:.3e}, "
            f"passed={success}"
        )

    print(f"\nGradient check passed: {passed}/{checks_done}")
    print("Requirement satisfied:", checks_done == num_checks and passed >= 9)

print("\nNumerical gradient checking")

check_parameters = initialize_parameters(16)

gradient_check(
    X_train[:16],
    Y_train[:16],
    check_parameters
)


# Task 6

import time
import matplotlib.pyplot as plt


def train_network(width, learning_rate, epochs=50, batch_size=32):
    parameters = initialize_parameters(width, seed=42)
    n_samples = X_train.shape[0]
    rng = np.random.default_rng(42)

    epoch_losses = []

    for epoch in range(epochs):
        indices = rng.permutation(n_samples)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]

        batch_losses = []

        for start in range(0, n_samples, batch_size):
            X_batch = X_shuffled[start:start + batch_size]
            Y_batch = Y_shuffled[start:start + batch_size]

            probabilities, cache = forward_propagation(
                X_batch, parameters
            )

            loss = cross_entropy_loss(Y_batch, probabilities)
            gradients = backward_propagation(
                Y_batch, parameters, cache
            )

            W1, b1, W2, b2, W3, b3 = parameters
            dW1, db1, dW2, db2, dW3, db3 = gradients

            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            W3 -= learning_rate * dW3
            b3 -= learning_rate * db3

            batch_losses.append(loss)

        epoch_loss = np.mean(batch_losses)
        epoch_losses.append(epoch_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Width={width}, LR={learning_rate}, "
                f"Epoch={epoch + 1}/{epochs}, "
                f"Loss={epoch_loss:.6f}"
            )

    test_probabilities, _ = forward_propagation(
        X_test, parameters
    )
    predictions = np.argmax(test_probabilities, axis=1)
    test_accuracy = np.mean(predictions == y_test)

    return parameters, epoch_losses, test_accuracy


learning_rates = [0.001, 0.01, 0.1]
widths = [16, 64]

results = {}

print("\nTask 6: Training experiments")

for width in widths:
    for learning_rate in learning_rates:
        print(f"\nStarting width={width}, LR={learning_rate}")

        start_time = time.perf_counter()

        parameters, losses, accuracy = train_network(
            width=width,
            learning_rate=learning_rate,
            epochs=50,
            batch_size=32
        )

        elapsed = time.perf_counter() - start_time

        results[(width, learning_rate)] = {
            "parameters": parameters,
            "losses": losses,
            "accuracy": accuracy,
            "time": elapsed
        }

        print(f"Final test accuracy: {accuracy:.4f}")
        print(f"Training time: {elapsed:.2f} seconds")


plt.figure(figsize=(10, 6))

for (width, learning_rate), result in results.items():
    plt.plot(
        range(1, 51),
        result["losses"],
        label=f"W={width}, LR={learning_rate}"
    )

plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("Training Loss for Different Network Configurations")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lab02/results/loss_curves.png", dpi=150)
plt.close()

with open("lab02/results/experiment_results.txt", "w", encoding="utf-8") as file:
    file.write("Width,Learning Rate,Final Loss,Test Accuracy,Training Time (s)\n")

    for (width, learning_rate), result in results.items():
        file.write(
            f"{width},{learning_rate},"
            f"{result['losses'][-1]:.6f},"
            f"{result['accuracy']:.4f},"
            f"{result['time']:.2f}\n"
        )

print("\nAll experiments completed.")
print("Loss curves saved to lab02/results/loss_curves.png")
print("Results saved to lab02/results/experiment_results.txt")



# Task 7

def benchmark_network(width, runs=100):
    parameters = results[(width, 0.1)]["parameters"]

    X_batch = X_train[:32]
    Y_batch = Y_train[:32]

    forward_times = []
    backward_times = []

    for _ in range(10):
        _, cache = forward_propagation(X_batch, parameters)
        backward_propagation(Y_batch, parameters, cache)

    for _ in range(runs):
        start = time.perf_counter()
        _, cache = forward_propagation(X_batch, parameters)
        forward_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        backward_propagation(Y_batch, parameters, cache)
        backward_times.append(time.perf_counter() - start)

    median_forward = np.median(forward_times)
    median_backward = np.median(backward_times)

    print(f"\nTiming results for width={width}")
    print(f"Forward median:  {median_forward * 1000:.4f} ms")
    print(f"Backward median: {median_backward * 1000:.4f} ms")

    if median_forward > 0:
        ratio = median_backward / median_forward
        print(f"Backward / Forward ratio: {ratio:.2f}x")

    return median_forward, median_backward


print("\nTask 7: Timing benchmark (100 runs)")

timing_results = {}

for width in [16, 64]:
    timing_results[width] = benchmark_network(width, runs=100)

