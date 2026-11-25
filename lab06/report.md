# Lab 6 — Training Pipelines and Project Proposal

## 1. Model selection and training

The three architectures from Lab 3 (MLP, CNN, and Attention) were evaluated on the Fashion-MNIST dataset.

The CNN architecture achieved the best validation performance and was selected for the final training pipeline.

- Selected architecture: CNN
- Number of parameters: 20,490
- Final test accuracy: 89.89%

The training pipeline uses a YAML configuration file, fixed random seeds, checkpoint saving, and MLflow experiment tracking.

## 2. PyTorch profiling

The selected CNN model was profiled using `torch.profiler`.

| Operation | CPU time share |
|---|---:|
| Data loading (DataLoader) | 64.11% |
| Forward pass | 13.57% |
| Backward pass | 20.37% |
| Optimizer step | 1.79% |

Data loading is the main performance bottleneck, accounting for approximately 64% of the measured CPU time. Forward and backward passes account for approximately 34% combined.

The profiler output and Chrome trace are available in:
- `results/torch_profiler_table.txt`
- `results/torch_profile_trace.json`

## 3. DataLoader worker study

The number of DataLoader workers was varied across 0, 1, 2, 4, and 8.

| Workers | Time (s) | Throughput (samples/s) |
|---:|---:|---:|
| 0 | 4.084 | 2006.0 |
| 1 | 15.417 | 531.4 |
| 2 | 24.563 | 333.5 |
| 4 | 46.967 | 174.4 |
| 8 | 90.457 | 90.6 |

The best result was achieved with `num_workers=0`. Increasing the number of workers reduced throughput significantly. This suggests that multiprocessing overhead is more expensive than the benefits of parallel data loading in the tested environment.

## 4. Batch-size study

Batch sizes of 32, 64, 128, 256, and 512 were evaluated over five epochs.

| Batch size | Epochs | Throughput (samples/s) | Peak process memory (MB) | Test accuracy |
|---:|---:|---:|---:|---:|
| 32 | 5 | 955.8 | 147.0 | 0.9047 |
| 64 | 5 | 1030.3 | 130.6 | 0.8991 |
| 128 | 5 | 1294.5 | 144.4 | 0.8981 |
| 256 | 5 | 1353.0 | 174.4 | 0.8914 |
| 512 | 5 | 597.4 | 220.3 | 0.8858 |

Batch size 256 achieved the highest throughput, while batch size 32 achieved the highest test accuracy. Batch size 64 used the least memory. Batch size 512 performed poorly in both throughput and accuracy.

Therefore, the best batch size depends on whether throughput, memory consumption, or predictive accuracy is the primary objective.

## 5. Optimizer and learning-rate scheduler comparison

SGD with momentum and Adam were evaluated with StepLR, CosineAnnealingLR, and ReduceLROnPlateau. Each configuration was trained for ten epochs.

| Optimizer | Scheduler | Epochs | Test accuracy | Test loss | Final LR |
|---|---|---:|---:|---:|---:|
| SGD | StepLR | 10 | 0.8627 | 0.3938 | 1.25e-04 |
| SGD | CosineAnnealingLR | 10 | 0.8600 | 0.3954 | 2.45e-05 |
| SGD | ReduceLROnPlateau | 10 | 0.8663 | 0.3681 | 1.00e-03 |
| Adam | StepLR | 10 | 0.9048 | 0.2706 | 1.25e-04 |
| Adam | CosineAnnealingLR | 10 | 0.9051 | 0.2702 | 2.45e-05 |
| Adam | ReduceLROnPlateau | 10 | 0.9009 | 0.2797 | 1.00e-03 |

Adam outperformed SGD in all three scheduler configurations. Adam with CosineAnnealingLR achieved the highest test accuracy of 90.51%, although its difference from Adam with StepLR was very small.

## 6. Gradient accumulation

Gradient accumulation was used to emulate an effective batch size of 256 by processing four micro-batches of 64.

The final validation accuracies were:

- Batch size 64 with accumulation of 4 steps: 0.8900
- Physical batch size 256: 0.8902

The results are very close, indicating that gradient accumulation successfully approximated training with a larger physical batch size in this experiment.

Detailed training histories are available in `results/gradient_accumulation.json`.

## 7. Checkpointing and training resumption

Training was interrupted after epoch 3 and resumed from a checkpoint containing the model and optimizer states.

After resuming, training continued through epoch 10. The final validation accuracy reached 0.9070.

This confirms that the training pipeline can restore the saved state and continue training rather than restarting from the beginning.

Detailed results are available in `results/resume_training.json`.

## 8. MLflow experiment tracking

The experiment was logged using MLflow under the experiment name `Lab06_Training_Pipelines`.

The run recorded:
- Model architecture
- Batch size
- Number of epochs
- Random seed
- Number of model parameters
- Training and validation metrics
- Test accuracy and test loss
- Model checkpoint artifact

The local MLflow tracking data is stored in the `mlruns` directory.

## 9. Individual project proposal

The individual project proposal is provided in `individual_project_proposal.md`.

## 10. Conclusions

1. The CNN architecture achieved the best validation performance among the evaluated Lab 3 architectures.
2. Data loading was the main CPU bottleneck, accounting for 64.11% of the profiler's CPU time.
3. Increasing DataLoader workers reduced throughput in the tested environment; `num_workers=0` was the best configuration.
4. Batch size 256 provided the highest throughput, while batch size 32 achieved the highest test accuracy.
5. Adam performed better than SGD in all tested scheduler configurations. Adam with CosineAnnealingLR achieved the highest test accuracy.
6. Gradient accumulation produced results very close to those obtained with a physical batch size of 256.
7. Checkpointing successfully restored both model and optimizer states, allowing training to resume from epoch 3.
8. YAML configuration, reproducible seeds, checkpointing, and MLflow tracking provide a more reproducible and manageable training pipeline.