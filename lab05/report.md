# Lab 5: Frameworks and Automatic Differentiation

**Course:** Machine Learning  
**Laboratory:** Lab 5 — Frameworks and Automatic Differentiation  
**Student:** Bohdan 72612  
**Environment:** Python 3.11, PyTorch 2.3.1+cpu

---

## 1. Introduction

The purpose of this laboratory work was to understand the fundamental concepts behind automatic differentiation and machine learning frameworks.

The main objectives were:

- Implement a scalar automatic differentiation engine.
- Construct computation graphs during the forward pass.
- Implement backpropagation using topological sorting.
- Verify calculated gradients against PyTorch Autograd and numerical approximations.
- Train a small neural network using a custom implementation.
- Compare custom engine performance with PyTorch.
- Investigate eager and traced execution modes.
- Analyze the performance impact of contiguous and non-contiguous tensors.

## 2. Task 1 — Value Node Implementation

The first task was to design a `Value` class representing a node in a computational graph.

Each node stores:

- `data` — the numerical value.
- `grad` — the accumulated gradient.
- `_prev` — references to parent nodes.
- `_op` — the operation that created the node.
- `_backward` — a function implementing the local derivative rule.

The class allows individual scalar values to become part of a computational graph.

Each operation creates a new `Value` object and stores references to its parent nodes.

This approach allows the engine to reconstruct the computational graph and calculate gradients during backpropagation.

## 3. Task 2 — Arithmetic Operations and Computation Graph

The following operations were implemented:

- Addition (`+`)
- Multiplication (`*`)
- Power (`**`)
- ReLU activation

The implementation uses the chain rule to calculate local derivatives.

### 3.1 Addition

For:

`c = a + b`

The local derivatives are:

\[
\frac{\partial c}{\partial a}=1,\qquad
\frac{\partial c}{\partial b}=1
\]

### 3.2 Multiplication

For:

`c = a * b`

The local derivatives are:

\[
\frac{\partial c}{\partial a}=b,\qquad
\frac{\partial c}{\partial b}=a
\]

### 3.3 Power

For:

`f(x) = x^n`

The derivative is:

\[
f'(x)=nx^{n-1}
\]

### 3.4 ReLU

The ReLU activation function is:

\[
ReLU(x)=\max(0,x)
\]

Its derivative is:

\[
ReLU'(x)=
\begin{cases}
1 & x>0\\
0 & x\leq 0
\end{cases}
\]

The forward operations automatically construct a computation graph by storing their parent nodes and local backward functions.

## 4. Task 3 — Backpropagation

A backward pass was implemented using topological sorting.

The algorithm consists of the following steps:

1. Traverse the computational graph recursively.
2. Build a topologically sorted list of nodes.
3. Set the output gradient to 1.
4. Traverse the nodes in reverse topological order.
5. Execute the local backward functions.
6. Accumulate gradients for parent nodes.

Gradient accumulation is implemented using `+=` instead of assignment.

This is important because a single node may contribute to the final result through multiple computational paths.

### 4.1 Backpropagation Verification

The following expression was tested:

\[
f=(a+b)\cdot a
\]

With:

\[
a=3,\qquad b=4
\]

The expected result is:

\[
f=(3+4)\cdot3=21
\]

The analytical gradients are:

\[
\frac{\partial f}{\partial a}=2a+b=10
\]

\[
\frac{\partial f}{\partial b}=a=3
\]

Obtained results:

| Parameter | Expected | Obtained |
|---|---:|---:|
| Output | 21 | 21 |
| Gradient of a | 10 | 10 |
| Gradient of b | 3 | 3 |

The results confirmed that topological backpropagation and gradient accumulation worked correctly.

## 5. Task 4 — Gradient Verification

The custom automatic differentiation engine was compared with PyTorch Autograd and numerical gradient approximations.

The tested expression was:

\[
f(a,b,c)=ReLU((a\cdot b+c)^2)
\]

Input values:

\[
a=2,\qquad b=3,\qquad c=1
\]

The output was:

\[
f=(2\cdot3+1)^2=49
\]

The analytical gradients were:

\[
\frac{\partial f}{\partial a}=42
\]

\[
\frac{\partial f}{\partial b}=28
\]

\[
\frac{\partial f}{\partial c}=14
\]

### 5.1 Comparison Results

| Variable | Custom Engine | PyTorch | Numerical Gradient |
|---|---:|---:|---:|
| a | 42.0 | 42.0 | 41.999999997 |
| b | 28.0 | 28.0 | 28.000000004 |
| c | 14.0 | 14.0 | 14.000000004 |

The maximum absolute difference was:

\[
3.9138\times10^{-9}
\]

The required tolerance was:

\[
10^{-6}
\]

**Result: PASSED**

The maximum difference was significantly smaller than the required tolerance.

This confirms that the custom engine correctly calculates gradients for the tested expression.

## 6. Task 5 — Neural Network Training on make_moons

A small neural network was implemented and trained using the custom automatic differentiation engine.

### 6.1 Dataset

The `make_moons` dataset from Scikit-learn was used.

Configuration:

| Parameter | Value |
|---|---:|
| Number of samples | 200 |
| Number of input features | 2 |
| Number of classes | 2 |
| Noise | 0.1 |
| Random state | 42 |

The dataset contains two interleaving moon-shaped classes and is suitable for binary classification experiments.

### 6.2 Network Architecture

The implemented network architecture was:

\[
2\rightarrow8\rightarrow1
\]

The architecture consists of:

- Input layer: 2 features.
- Hidden layer: 8 neurons with ReLU activation.
- Output layer: 1 neuron with Sigmoid activation.

The network contains 33 trainable parameters:

| Layer | Parameters |
|---|---:|
| Input to hidden | 24 |
| Hidden to output | 9 |
| **Total** | **33** |

Binary Cross Entropy (BCE) was used as the loss function.

The model was trained using gradient descent.

Training configuration:

| Parameter | Value |
|---|---:|
| Epochs | 150 |
| Learning rate | 0.05 |
| Optimizer | Gradient Descent |
| Loss function | Binary Cross Entropy |

### 6.3 Training Results

The training loss decreased throughout the experiment.

| Epoch | Loss |
|---|---:|
| 25 | 0.6869 |
| 50 | 0.5624 |
| 75 | 0.4827 |
| 100 | 0.4275 |
| 125 | 0.3893 |
| 150 | 0.3618 |

Final training accuracy:

\[
Accuracy=0.84=84\%
\]

The decreasing loss indicates that the network successfully learned patterns in the dataset.

The reported accuracy was measured on the same dataset used for training, so it represents training accuracy rather than independent test performance.


### 6.4 Custom Engine vs PyTorch Performance

The custom scalar engine was compared with an equivalent PyTorch neural network.

Both implementations used:
- The same make_moons dataset.
- The same network architecture: 2 → 8 → 1.
- ReLU in the hidden layer.
- Sigmoid in the output layer.
- Binary Cross Entropy loss.
- 150 training epochs.
- Five benchmark runs.

The median execution time was used for comparison.

| Implementation | Median Training Time |
|---|---:|
| Custom Autograd Engine | 59.7182 seconds |
| PyTorch | 0.2409 seconds |

The measured speedup was:

\[
Speedup = \frac{59.7182}{0.2409} \approx 247.86
\]

PyTorch was approximately 247.86 times faster in this experiment.

The difference can be explained by the implementation approach. The custom engine performs operations using individual Python scalar objects and processes samples one by one, while PyTorch uses optimized tensor operations and batch processing.



## 7. Task 6 — Execution Modes and Tensor Layout

The final task investigated PyTorch execution modes and the performance impact of tensor memory layout.

PyTorch eager execution was compared with TorchScript traced execution using torch.jit.trace.

### 7.1 Model Configuration

The model consisted of:
- Linear(64, 32)
- ReLU
- Linear(32, 10)

Input tensor:
- Shape: 128 × 64
- Data type: float32
- Device: CPU

The benchmark used 20 warm-up iterations, five measurement runs, and 100 iterations per run.

### 7.2 Eager vs Traced Execution

| Execution Mode | Median Time |
|---|---:|
| Eager Execution | 0.1148 ms |
| Traced Execution | 0.1047 ms |

The measured speedup was:

\[
Speedup = \frac{0.1148}{0.1047} \approx 1.097
\]

Traced execution was approximately 1.097 times faster than eager execution in this experiment.

### 7.3 Tensor Layout Investigation

Two tensors with the same logical shape were compared.

| Property | Contiguous | Non-contiguous |
|---|---|---|
| Shape | 1024 × 64 | 1024 × 64 |
| Dtype | float32 | float32 |
| Contiguous | True | False |
| Stride | (64, 1) | (128, 2) |
| Median Time | 0.0985 ms | 0.1738 ms |

The non-contiguous tensor operation was approximately 1.76 times slower.

This demonstrates that memory layout and tensor strides can influence the performance of tensor operations.


## 8. Conclusions

During this laboratory work, a scalar automatic differentiation engine was successfully implemented.

The implementation included:

- A `Value` class for storing numerical values and gradients.
- Automatic construction of computational graphs.
- Arithmetic operations and activation functions.
- Topological sorting and reverse-mode backpropagation.
- Gradient accumulation through multiple computational paths.

The custom engine was verified against PyTorch Autograd and numerical gradient estimates.

The maximum absolute gradient difference was approximately:

\[
3.91\times10^{-9}
\]

This satisfied the required tolerance of less than \(10^{-6}\).

A neural network with architecture 2 → 8 → 1 was successfully trained on the `make_moons` dataset, achieving 84% training accuracy.

Performance experiments demonstrated that PyTorch was significantly faster than the custom scalar implementation.

The final experiments also showed that traced execution was faster than eager execution for the tested model and that non-contiguous tensors required more time for the tested matrix multiplication operation.

Overall, the laboratory provided practical experience with automatic differentiation, computational graphs, backpropagation, neural network training, and the internal execution mechanisms of machine learning frameworks.

## 9. Limitations

Several limitations should be considered:

- The custom engine operates on scalar values rather than optimized tensors.
- The neural network accuracy was measured on the training dataset.
- Performance measurements were performed on a CPU-only environment.
- Benchmark results may vary depending on hardware and system load.
- The experiments used relatively small datasets and models.

These limitations should be considered when interpreting the results and comparing the implementations.

---

**End of Laboratory Report**