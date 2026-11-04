
import random
from sklearn.datasets import make_moons
import json

from pathlib import Path
from model import MLP, binary_cross_entropy
from value import Value


# Reproducibility
random.seed(42)

# Dataset
X, y = make_moons(
    n_samples=200,
    noise=0.1,
    random_state=42
)

# Model: 2 -> 8 -> 1
model = MLP(2, 8, 1)

learning_rate = 0.05
epochs = 150

# Training
for epoch in range(epochs):

    # Reset parameter gradients
    for parameter in model.parameters():
        parameter.grad = 0.0

    # Forward pass
    losses = []

    for features, label in zip(X, y):
        prediction = model(features.tolist())
        loss = binary_cross_entropy(prediction, float(label))
        losses.append(loss)

    # Mean loss
    total_loss = sum(losses, Value(0)) / len(losses)

    # Backward pass
    total_loss.backward()

    # Gradient descent
    for parameter in model.parameters():
        parameter.data -= learning_rate * parameter.grad

    if (epoch + 1) % 25 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {total_loss.data:.4f}"
        )


# Evaluation
correct = 0

for features, label in zip(X, y):
    prediction = model(features.tolist())

    predicted_class = 1 if prediction.data >= 0.5 else 0

    if predicted_class == label:
        correct += 1

accuracy = correct / len(y)

print("\nTraining completed!")
print(f"Final accuracy: {accuracy:.4f}")
print(f"Final accuracy (%): {accuracy * 100:.2f}%")


# Save training results
results_dir = Path(__file__).resolve().parents[1] / "results"
results_dir.mkdir(exist_ok=True)

training_results = {
    "dataset": "make_moons",
    "samples": 200,
    "features": 2,
    "classes": 2,
    "architecture": "2-8-1",
    "parameters": len(model.parameters()),
    "epochs": epochs,
    "learning_rate": learning_rate,
    "final_loss": total_loss.data,
    "training_accuracy": accuracy,
    "training_accuracy_percent": accuracy * 100
}

with open(results_dir / "training_results.json", "w") as file:
    json.dump(training_results, file, indent=4)

print("Training results saved to results/training_results.json")