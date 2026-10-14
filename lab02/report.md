# Lab 2: A Neural Network from Scratch

## 1. Objective

The objective of this lab was to implement a two-hidden-layer neural network from scratch using NumPy. The implementation includes forward propagation, softmax, cross-entropy loss, backpropagation, numerical gradient checking, training experiments, and timing measurements.

## 2. Dataset and Preprocessing

The `load_digits` dataset from scikit-learn was used. Each sample contains 64 features representing an 8 × 8 image of a handwritten digit.

Preprocessing steps:
- Input values were converted to `float64` and scaled by dividing by 16.
- Labels were one-hot encoded into 10 classes.
- The dataset was split into 70% training data and 30% test data.
- The split used `random_state=42` and stratification by label.

Dataset dimensions:

| Dataset | Shape |
|---|---:|
| Full input matrix `X` | `(1797, 64)` |
| One-hot labels `Y` | `(1797, 10)` |
| Training inputs | `(1257, 64)` |
| Test inputs | `(540, 64)` |
| Training labels | `(1257, 10)` |
| Test labels | `(540, 10)` |

The normalized input range was from 0.0 to 1.0.

## 3. Network Architecture

The network has 64 input features, two hidden layers of equal width, and 10 output classes.

The two hidden widths tested were `W = 16` and `W = 64`.

| Parameter | Shape when `W = 16` | Shape when `W = 64` |
|---|---|---|
| `W1` | `(64, 16)` | `(64, 64)` |
| `b1` | `(16,)` | `(64,)` |
| `W2` | `(16, 16)` | `(64, 64)` |
| `b2` | `(16,)` | `(64,)` |
| `W3` | `(16, 10)` | `(64, 10)` |
| `b3` | `(10,)` | `(10,)` |

Weights were initialized from a normal distribution with mean 0 and standard deviation 0.01. Biases were initialized to zero.

The activation function for each hidden layer was ReLU. Softmax was used at the output layer to produce class probabilities.

## 4. Forward Propagation

For input matrix `X`, the forward pass is:

`Z1 = X @ W1 + b1`

`A1 = ReLU(Z1)`

`Z2 = A1 @ W2 + b2`

`A2 = ReLU(Z2)`

`Z3 = A2 @ W3 + b3`

`P = Softmax(Z3)`

The output shape for a batch of 32 samples was `(32, 10)`. The sum of the output probabilities for each sample was approximately 1, with the check passing at a tolerance of `1e-6`.

## 5. Cross-Entropy Loss

The cross-entropy loss was implemented as:

`L = -(1/N) * sum(Y * log(P))`

A small clipping value was used to avoid taking the logarithm of zero.

For an untrained network, the observed losses were:

| Hidden width | Initial loss |
|---:|---:|
| 16 | 2.302586 |
| 64 | 2.302592 |

These values are close to `ln(10) ≈ 2.302585`, which is expected when the initial predictions are close to uniform across the 10 classes.

## 6. Backpropagation and Gradient Checking

Backpropagation was implemented manually using the chain rule. The gradients were computed for all three weight matrices and all three bias vectors.

The gradient arrays had the expected shapes, and all gradient values were finite.

Numerical gradient checking compared analytical gradients against central finite differences using `epsilon = 1e-5`. The final check reported **9 out of 10 checks passed**, satisfying the lab requirement of at least 9 of 10 weights with relative error below `1e-6`.

The checker used weights whose ReLU activation masks remained stable under the positive and negative perturbations, avoiding finite-difference estimates that cross ReLU's non-differentiable point.

## 7. Training Experiments

The network was trained for 50 epochs using mini-batches of 32 samples. The training samples were shuffled each epoch. The learning rates tested were `0.001`, `0.01`, and `0.1`, for both hidden widths.

| Hidden width | Learning rate | Final training loss | Test accuracy | Training time (s) |
|---:|---:|---:|---:|---:|
| 16 | 0.001 | 2.302571 | 10.19% | 0.44 |
| 16 | 0.01 | 2.302648 | 10.19% | 0.47 |
| 16 | 0.1 | 1.564644 | 32.59% | 0.55 |
| 64 | 0.001 | 2.302547 | 10.19% | 0.68 |
| 64 | 0.01 | 2.302517 | 10.19% | 0.72 |
| 64 | 0.1 | 0.158629 | 93.89% | 0.67 |

The loss curves were saved to `results/loss_curves.png`, and the experiment table was saved to `results/experiment_results.txt`.

### Discussion

The best result was obtained with hidden width 64 and learning rate 0.1, achieving 93.89% test accuracy and a final training loss of 0.158629.

For learning rates 0.001 and 0.01, both widths remained near 10.19% test accuracy after 50 epochs. This indicates that these settings did not produce effective learning within the chosen training duration. With learning rate 0.1, the wider network learned much more effectively than the narrower network in this experiment.

## 8. Timing Measurements

Forward and backward propagation were measured separately using 100 runs on a batch of 32 samples. The median runtime was reported. Ten warm-up runs were performed before measurements.

| Hidden width | Forward median (ms) | Backward median (ms) | Backward / Forward |
|---:|---:|---:|---:|
| 16 | 0.0885 | 0.1068 | 1.21× |
| 64 | 0.1258 | 0.1702 | 1.35× |

Backward propagation was slower than forward propagation for both widths. It computes gradients for the weights and biases and performs additional matrix multiplications. The wider network also required more computation, resulting in higher measured median times.

The timing results were saved to `results/timing_results.txt`.

## 9. Conclusion

In this lab, a two-hidden-layer neural network was implemented from scratch using NumPy. The implementation covered dataset preparation, parameter initialization, forward propagation, softmax, cross-entropy loss, backpropagation, numerical gradient checking, training, and timing analysis.

The output dimensions and softmax probability sums were validated. The gradient check passed 9 out of 10 tested weights, meeting the stated threshold. Six training configurations were compared, and the best performance was achieved by the network with width 64 and learning rate 0.1, reaching 93.89% test accuracy. Timing measurements showed that backward propagation took longer than forward propagation for both network widths.

Overall, the experiments demonstrated how hidden-layer width and learning rate affect learning performance, and how matrix operations contribute to neural-network computation time.
