import torch
from torch import nn

class FashionCNN(nn.Module):
    def __init__(self, channels=(32, 64, 128), hidden=512, num_classes=10):
        super().__init__()
        a, b, c = map(int, channels)
        self.conv1 = nn.Conv2d(1, a, 3, padding=1)
        self.conv2 = nn.Conv2d(a, b, 3, padding=1)
        self.conv3 = nn.Conv2d(b, c, 3, padding=1)
        self.relu, self.pool = nn.ReLU(inplace=True), nn.MaxPool2d(2)
        self.adaptive = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1, self.fc2 = nn.Linear(c * 16, hidden), nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return self.fc2(self.relu(self.fc1(self.adaptive(x).flatten(1))))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def build_baseline(cfg):
    return FashionCNN(cfg["model"]["baseline_channels"], cfg["model"]["baseline_hidden"])

def build_student(cfg):
    return FashionCNN(cfg["model"]["student_channels"], cfg["model"]["student_hidden"])
