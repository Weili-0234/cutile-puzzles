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

# Allow running from any directory
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

HINT: mask = ct.greater(a_tile, 0); result = ct.where(mask, a_tile, 0.0)
"""


def ref_relu(A: torch.Tensor) -> torch.Tensor:
    return torch.relu(A)


@ct.kernel
def ct_relu(a, c, TILE: ConstInt):
    bid = ct.bid(0)

    # Compute global indices for this block's tile
    indices = bid * TILE + ct.arange(TILE, dtype=ct.int32)

    # TODO: Implement ReLU using gather/scatter
    # 1. Gather elements from array a at indices
    # 2. Create a boolean mask: ct.greater(a_tile, 0)
    # 3. Apply ReLU: ct.where(mask, a_tile, 0.0)
    # 4. Scatter the result to array c at indices
    pass


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
    test_puzzle(
        launch_relu,
        ref_relu,
        inputs,
        label="02-1 ReLU (gather/scatter)",
        hint="mask = ct.greater(a_tile, 0); result = ct.where(mask, a_tile, 0.0)",
    )


# ---------------------------------------------------------------------------
# 02-2: Fused Mul + ReLU — C[i] = max(0, A[i] * B[i])
# ---------------------------------------------------------------------------
r"""
Combine multiplication and ReLU into a single fused kernel.

C[i] = max(0, A[i] * B[i])

This demonstrates the power of kernel fusion: performing multiple operations
in a single pass over the data to reduce memory traffic.

Inputs:
    A: Tensor([N,], float32)
    B: Tensor([N,], float32)

Output:
    C: Tensor([N,], float32)  where C[i] = max(0, A[i] * B[i])

HINT: Multiply first, then apply ReLU with ct.where and ct.greater.
"""


def ref_fused_mul_relu(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.relu(A * B)


@ct.kernel
def ct_fused_mul_relu(a, b, c, TILE: ConstInt):
    bid = ct.bid(0)

    # Compute global indices for this block's tile
    indices = bid * TILE + ct.arange(TILE, dtype=ct.int32)

    # TODO: Implement fused multiply + ReLU using gather/scatter
    # 1. Gather elements from a and b at indices
    # 2. Multiply: product = a_tile * b_tile
    # 3. Create boolean mask: ct.greater(product, 0)
    # 4. Apply ReLU: ct.where(mask, product, 0.0)
    # 5. Scatter the result to c at indices
    pass


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
        hint="Multiply a_tile * b_tile first, then apply ct.where(ct.greater(product, 0), product, 0.0).",
    )


# ---------------------------------------------------------------------------
# 02-3: SiLU (Sigmoid Linear Unit) — C[i] = A[i] * sigmoid(A[i])
# ---------------------------------------------------------------------------
r"""
Implement the SiLU (Sigmoid Linear Unit) activation, also known as "Swish".

SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))

This sub-task introduces `@ct.function`, which lets you define reusable helper
functions that can be called from within kernels.

sigmoid(x) = 1.0 / (1.0 + ct.exp(-x))

Define sigmoid as a @ct.function, then use it inside the kernel.

Inputs:
    A: Tensor([N,], float32)

Output:
    C: Tensor([N,], float32)  where C[i] = A[i] * sigmoid(A[i])

HINT: Define `@ct.function` for sigmoid: `return 1.0 / (1.0 + ct.exp(-x))`.
      Then compute `a_tile * sigmoid(a_tile)`.
"""


def ref_silu(A: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(A)


# TODO: Define a @ct.function helper for sigmoid
# @ct.function
# def sigmoid(x):
#     return 1.0 / (1.0 + ct.exp(-x))


@ct.kernel
def ct_silu(a, c, TILE: ConstInt):
    bid = ct.bid(0)

    # TODO: Implement SiLU using load/store
    # 1. Load a tile of size TILE from array a at index (bid,)
    # 2. Compute sigmoid: sig = sigmoid(a_tile)  (use the @ct.function helper)
    # 3. Compute SiLU: result = a_tile * sig
    # 4. Store the result to array c at index (bid,)
    pass


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
    test_puzzle(
        launch_silu,
        ref_silu,
        inputs,
        label="02-3 SiLU (load/store)",
        hint="Define @ct.function sigmoid(x): return 1.0 / (1.0 + ct.exp(-x)). "
        "Then silu = a_tile * sigmoid(a_tile).",
    )


if __name__ == "__main__":
    run_relu()
    run_fused_mul_relu()
    run_silu()
