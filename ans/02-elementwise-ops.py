"""
Puzzle 02: Element-wise Operations
====================================
Apply common element-wise operations using cutile.
Learn `ct.where`, `ct.greater`, `ct.exp`, and `@ct.function` helpers.

Category: ["official"]
Difficulty: ["easy"]
"""

import math
import os
import sys

import cuda.tile as ct
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.utils import bench_puzzle, test_puzzle

# Type alias for compile-time constants
ConstInt = ct.Constant[int]


# ---------------------------------------------------------------------------
# 02-1: ReLU — C[i] = max(0, A[i])
# ---------------------------------------------------------------------------
r"""
Apply the ReLU (Rectified Linear Unit) activation function element-wise.

ReLU(x) = max(0, x) = x if x > 0 else 0

Use `ct.greater(tile, 0)` to produce a boolean mask, and
`ct.where(mask, true_val, false_val)` to select between two values.

Inputs:
    A: Tensor([N,], float32)

Output:
    C: Tensor([N,], float32)  where C[i] = max(0, A[i])
"""


def ref_relu(A: torch.Tensor) -> torch.Tensor:
    return torch.relu(A)


@ct.kernel
def ct_relu(a, c, TILE: ConstInt):
    bid = ct.bid(0)

    # Compute global indices for this block's tile
    indices = bid * TILE + ct.arange(TILE, dtype=ct.int32)

    # Gather elements from array a
    a_tile = ct.gather(a, indices)

    # Create boolean mask: True where a > 0
    mask = ct.greater(a_tile, 0)

    # Apply ReLU: keep positive values, zero out negatives
    result = ct.where(mask, a_tile, 0.0)

    # Scatter the result back
    ct.scatter(c, indices, result)


def launch_relu(A: torch.Tensor) -> torch.Tensor:
    N = A.shape[0]
    TILE = 1024
    C = torch.empty_like(A)
    grid = (math.ceil(N / TILE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, ct_relu, (A, C, TILE))
    return C


def run_relu():
    print("\n=== 02-1: ReLU (gather/scatter) ===\n")
    N = 8192
    inputs = {
        "A": torch.randn(N, dtype=torch.float32, device="cuda"),
    }
    test_puzzle(launch_relu, ref_relu, inputs, label="02-1 ReLU (gather/scatter)")
    bench_puzzle(launch_relu, ref_relu, inputs, bench_torch=True)


# ---------------------------------------------------------------------------
# 02-2: Fused Mul + ReLU — C[i] = max(0, A[i] * B[i])
# ---------------------------------------------------------------------------
r"""
Combine multiplication and ReLU into a single fused kernel.

C[i] = max(0, A[i] * B[i])

Inputs:
    A: Tensor([N,], float32)
    B: Tensor([N,], float32)

Output:
    C: Tensor([N,], float32)  where C[i] = max(0, A[i] * B[i])
"""


def ref_fused_mul_relu(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.relu(A * B)


@ct.kernel
def ct_fused_mul_relu(a, b, c, TILE: ConstInt):
    bid = ct.bid(0)

    # Compute global indices for this block's tile
    indices = bid * TILE + ct.arange(TILE, dtype=ct.int32)

    # Gather elements from both arrays
    a_tile = ct.gather(a, indices)
    b_tile = ct.gather(b, indices)

    # Fused multiply
    product = a_tile * b_tile

    # Apply ReLU
    mask = ct.greater(product, 0)
    result = ct.where(mask, product, 0.0)

    # Scatter the result back
    ct.scatter(c, indices, result)


def launch_fused_mul_relu(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    N = A.shape[0]
    TILE = 1024
    C = torch.empty_like(A)
    grid = (math.ceil(N / TILE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, ct_fused_mul_relu, (A, B, C, TILE))
    return C


def run_fused_mul_relu():
    print("\n=== 02-2: Fused Mul + ReLU (gather/scatter) ===\n")
    N = 8192
    inputs = {
        "A": torch.randn(N, dtype=torch.float32, device="cuda"),
        "B": torch.randn(N, dtype=torch.float32, device="cuda"),
    }
    test_puzzle(
        launch_fused_mul_relu,
        ref_fused_mul_relu,
        inputs,
        label="02-2 Fused Mul+ReLU (gather/scatter)",
    )
    bench_puzzle(launch_fused_mul_relu, ref_fused_mul_relu, inputs, bench_torch=True)


# ---------------------------------------------------------------------------
# 02-3: SiLU (Sigmoid Linear Unit) — C[i] = A[i] * sigmoid(A[i])
# ---------------------------------------------------------------------------
r"""
Implement the SiLU (Sigmoid Linear Unit) activation, also known as "Swish".

SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))

Introduces `@ct.function` for defining reusable helper functions.

Inputs:
    A: Tensor([N,], float32)

Output:
    C: Tensor([N,], float32)  where C[i] = A[i] * sigmoid(A[i])
"""


def ref_silu(A: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(A)


@ct.function
def sigmoid(x):
    return 1.0 / (1.0 + ct.exp(-x))


@ct.kernel
def ct_silu(a, c, TILE: ConstInt):
    bid = ct.bid(0)

    # Load a tile from array a
    a_tile = ct.load(a, index=(bid,), shape=(TILE,))

    # Compute SiLU: x * sigmoid(x)
    sig = sigmoid(a_tile)
    result = a_tile * sig

    # Store the result
    ct.store(c, index=(bid,), tile=result)


def launch_silu(A: torch.Tensor) -> torch.Tensor:
    N = A.shape[0]
    TILE = 1024
    C = torch.empty_like(A)
    grid = (math.ceil(N / TILE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, ct_silu, (A, C, TILE))
    return C


def run_silu():
    print("\n=== 02-3: SiLU (load/store) ===\n")
    N = 8192
    inputs = {
        "A": torch.randn(N, dtype=torch.float32, device="cuda"),
    }
    test_puzzle(launch_silu, ref_silu, inputs, label="02-3 SiLU (load/store)")
    bench_puzzle(launch_silu, ref_silu, inputs, bench_torch=True)


if __name__ == "__main__":
    run_relu()
    run_fused_mul_relu()
    run_silu()
