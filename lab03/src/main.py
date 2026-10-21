
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchinfo import summary
from thop import profile


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3

DEVICE = torch.device("cpu")

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {DEVICE}")
print(f"Results directory: {RESULTS_DIR}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

full_train_dataset = datasets.FashionMNIST(
    root=BASE_DIR / "data",
    train=True,
    download=True,
    transform=transform
)

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [55000, 5000],
    generator=torch.Generator().manual_seed(SEED)
)

test_dataset = datasets.FashionMNIST(
    root=BASE_DIR / "data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

images, labels = next(iter(train_loader))
print(f"Batch image shape: {images.shape}")
print(f"Batch label shape: {labels.shape}")


# Class names in Fashion-MNIST
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


# 1. Multi-Layer Perceptron (MLP)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


# 2. Convolutional Neural Network (CNN)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_projection = nn.Linear(49, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, 4, 7, 4, 7)

        x = x.permute(0, 2, 4, 1, 3, 5)

        x = x.reshape(batch_size, 16, 49)

        x = self.patch_projection(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        return self.classifier(x)


models = {
    "MLP": MLP(),
    "CNN": CNN(),
    "Attention": AttentionModel(),
}

print("\nModel architecture checks:")

for name, model in models.items():
    model.eval()

    with torch.no_grad():
        output = model(images)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"{name}: output shape = {tuple(output.shape)}, "
        f"trainable parameters = {parameter_count:,}"
    )


print("\nModel profiling:")

for name, model in models.items():
    model.eval()

    print(f"\n--- {name} ---")

    summary(
        model,
        input_size=(1, 1, 28, 28),
        verbose=0,
    )

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    sample = torch.randn(1, 1, 28, 28)

    try:
        macs, thop_params = profile(
            model,
            inputs=(sample,),
            verbose=False,
        )

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"THOP parameters: {thop_params:,}")
        print(f"THOP MACs: {macs:,}")
        print(f"Approximate FLOPs (2 x MACs): {2 * macs:,}")

    except Exception as error:
        print(f"THOP profiling failed: {error}")
        print(f"Total parameters: {total_params:,}")


print("\nInference latency benchmark:")

def measure_latency(model, batch_size, runs):
    model.eval()
    sample = torch.randn(batch_size, 1, 28, 28, device=DEVICE)

    with torch.no_grad():
        for _ in range(10):
            model(sample)

        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            model(sample)
            end = time.perf_counter()
            timings.append((end - start) * 1000)

    return float(np.median(timings))


latency_results = {}

for name, model in models.items():
    latency_1 = measure_latency(model, batch_size=1, runs=100)
    latency_128 = measure_latency(model, batch_size=128, runs=20)

    latency_results[name] = {
        "batch_1_ms": latency_1,
        "batch_128_ms": latency_128,
    }

    print(f"\n{name}:")
    print(f"  Batch 1 median latency: {latency_1:.4f} ms")
    print(f"  Batch 128 median latency: {latency_128:.4f} ms")


import csv

print("\nModel training:")

training_results = {}

for name, model in models.items():
    print(f"\n{'=' * 50}")
    print(f"Training {name}")
    print(f"{'=' * 50}")

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    epoch_results = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)
                predictions = outputs.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total

        epoch_results.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.4%} | "
            f"Val acc: {val_accuracy:.4%}"
        )

    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)

    test_accuracy = test_correct / test_total

    training_results[name] = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "batch_1_ms": latency_results[name]["batch_1_ms"],
        "batch_128_ms": latency_results[name]["batch_128_ms"],
        "test_accuracy": test_accuracy,
        "epochs": epoch_results,
    }

    print(f"\n{name} final test accuracy: {test_accuracy:.4%}")

csv_path = RESULTS_DIR / "training_results.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "model", "parameters", "batch_1_ms",
            "batch_128_ms", "test_accuracy"
        ],
    )
    writer.writeheader()

    for name, result in training_results.items():
        writer.writerow({
            "model": name,
            "parameters": result["parameters"],
            "batch_1_ms": result["batch_1_ms"],
            "batch_128_ms": result["batch_128_ms"],
            "test_accuracy": result["test_accuracy"],
        })

print(f"\nTraining results saved to: {csv_path}")
