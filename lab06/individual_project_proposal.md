# Individual Project Proposal — Lightweight Fashion-MNIST Classifier

**Student:** Bohdan Nikolenko  
**Student ID:** 72612

## Problem
Build a compact image classifier for ten clothing categories that can run on a low-resource CPU device.

## Dataset
Fashion-MNIST: 60,000 official training images and 10,000 test images, 10 classes, 28×28 grayscale. Reserve 5,000 training examples for validation. Normalize using mean 0.2860 and standard deviation 0.3530.

## Success metric
Primary: test accuracy target ≥90%. Secondary: per-image CPU latency, model file size and peak memory.

## Deployment constraints
Latency ≤50 ms per image; serialized model size ≤5 MB; CPU inference without GPU.

## Planned optimization
Start from the best Lab 3 architecture selected by validation accuracy. Evaluate structured pruning and dynamic quantization where supported. Compare accuracy, latency and actual serialized size before/after optimization; retain an uncompressed baseline.

## Risks
CPU timing varies with system load. Use warm-up and repeated measurements. Quantization support depends on layer types, so validate the exported artifact instead of assuming theoretical savings.
