
import time
import statistics
import torch
import torch.nn as nn

import json
from pathlib import Path


torch.manual_seed(42)

print("PyTorch version:", torch.__version__)
print("Device: CPU")


# --------------------------------------------------
# 1. Create a small model
# --------------------------------------------------

model = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

model.eval()

sample = torch.randn(128, 64)


# --------------------------------------------------
# 2. Inspect model graph using TorchScript
# --------------------------------------------------

traced_model = torch.jit.trace(model, sample)

print("\nMODEL GRAPH:")
print(traced_model.graph)


# --------------------------------------------------
# 3. Eager vs Traced execution
# --------------------------------------------------

def benchmark(function, x, warmup=20, repeats=5, iterations=100):

    # Warm-up
    with torch.inference_mode():
        for _ in range(warmup):
            function(x)

    times = []

    for _ in range(repeats):
        start = time.perf_counter()

        with torch.inference_mode():
            for _ in range(iterations):
                function(x)

        elapsed = time.perf_counter() - start
        times.append(elapsed / iterations)

    return statistics.median(times)


eager_time = benchmark(model, sample)
traced_time = benchmark(traced_model, sample)

print("\nEXECUTION MODE RESULTS")
print(f"Eager median:  {eager_time * 1000:.4f} ms")
print(f"Traced median: {traced_time * 1000:.4f} ms")

if traced_time > 0:
    print(
        "Traced speedup:",
        f"{eager_time / traced_time:.3f}x"
    )


# --------------------------------------------------
# 4. Inspect tensor properties
# --------------------------------------------------

base = torch.randn(1024, 128)

contiguous_tensor = base[:, :64].contiguous()
non_contiguous_tensor = base[:, ::2]

print("\nTENSOR PROPERTIES")

for name, tensor in [
    ("Contiguous", contiguous_tensor),
    ("Non-contiguous", non_contiguous_tensor)
]:
    print(f"\n{name}:")
    print("Shape:", tensor.shape)
    print("Dtype:", tensor.dtype)
    print("Contiguous:", tensor.is_contiguous())
    print("Stride:", tensor.stride())


# --------------------------------------------------
# 5. Compare contiguous and non-contiguous operations
# --------------------------------------------------

weight = torch.randn(64, 32)

contiguous_input = contiguous_tensor


def tensor_operation(x):
    return x @ weight


contiguous_time = benchmark(
    tensor_operation,
    contiguous_input,
    iterations=500
)

non_contiguous_time = benchmark(
    tensor_operation,
    non_contiguous_tensor,
    iterations=500
)

print("\nTENSOR LAYOUT PERFORMANCE")
print(f"Contiguous median: {contiguous_time * 1000:.4f} ms")
print(f"Non-contiguous median: {non_contiguous_time * 1000:.4f} ms")

print("\nLab 5 execution experiments completed.")

# Save execution and tensor layout results
results_dir = Path(__file__).resolve().parents[1] / "results"
results_dir.mkdir(exist_ok=True)

execution_results = {
    "pytorch_version": torch.__version__,
    "device": "CPU",
    "execution_modes": {
        "eager_median_ms": eager_time * 1000,
        "traced_median_ms": traced_time * 1000,
        "traced_speedup": eager_time / traced_time
    },
    "tensor_layout": {
        "contiguous": {
            "shape": list(contiguous_tensor.shape),
            "dtype": str(contiguous_tensor.dtype),
            "is_contiguous": contiguous_tensor.is_contiguous(),
            "stride": list(contiguous_tensor.stride()),
            "median_ms": contiguous_time * 1000
        },
        "non_contiguous": {
            "shape": list(non_contiguous_tensor.shape),
            "dtype": str(non_contiguous_tensor.dtype),
            "is_contiguous": non_contiguous_tensor.is_contiguous(),
            "stride": list(non_contiguous_tensor.stride()),
            "median_ms": non_contiguous_time * 1000
        },
        "non_contiguous_slowdown": (
            non_contiguous_time / contiguous_time
        )
    }
}

with open(results_dir / "execution_results.json", "w") as file:
    json.dump(execution_results, file, indent=4)

print("Execution results saved to results/execution_results.json")