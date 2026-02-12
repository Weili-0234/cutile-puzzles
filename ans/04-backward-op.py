"""
Puzzle 04: Backward Op
=======================
Implement a forward and backward pass for a fused broadcast-multiply + ReLU operation.
Learn to compute gradients through element-wise and broadcast operations.

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
# 04-1: Forward — Broadcast Mul + ReLU: C[i, j] = max(0, A[i, j] * B[j])
# ---------------------------------------------------------------------------
r"""
Compute C[i, j] = max(0, A[i, j] * B[j])

B is a 1D vector that broadcasts across rows of A.

Inputs:
    A: Tensor([M, N], float32)
    B: Tensor([N,], float32)

Output:
    C: Tensor([M, N], float32)  where C[i, j] = max(0, A[i, j] * B[j])
"""


def ref_broadcast_mul_relu(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.relu(A * B[None, :])


@ct.kernel
def ct_broadcast_mul_relu(a, b, c, TILE_M: ConstInt, TILE_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    # Create index tiles
    row_indices = bid_m * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    col_indices = bid_n * TILE_N + ct.arange(TILE_N, dtype=ct.int32)

    # Gather 2D tile from A using broadcasting indices
    a_tile = ct.gather(a, (row_indices[:, None], col_indices[None, :]))

    # Gather 1D tile from B and broadcast across rows
    b_tile = ct.gather(b, col_indices)
    b_broadcast = b_tile[None, :]  # shape (1, TILE_N)

    # Broadcast multiply
    product = a_tile * b_broadcast

    # Apply ReLU
    mask = ct.greater(product, 0)
    result = ct.where(mask, product, 0.0)

    # Scatter result to C
    ct.scatter(c, (row_indices[:, None], col_indices[None, :]), result)


def launch_broadcast_mul_relu(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, N = A.shape
    TILE_M, TILE_N = 32, 32
    C = torch.empty_like(A)
    grid = (math.ceil(M / TILE_M), math.ceil(N / TILE_N), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        ct_broadcast_mul_relu,
        (A, B, C, TILE_M, TILE_N),
    )
    return C


def run_broadcast_mul_relu():
    print("\n=== 04-1: Broadcast Mul + ReLU (gather/scatter) ===\n")
    M, N = 256, 512
    inputs = {
        "A": torch.randn(M, N, dtype=torch.float32, device="cuda"),
        "B": torch.randn(N, dtype=torch.float32, device="cuda"),
    }
    test_puzzle(
        launch_broadcast_mul_relu,
        ref_broadcast_mul_relu,
        inputs,
        label="04-1 Broadcast Mul+ReLU (gather/scatter)",
    )
    bench_puzzle(
        launch_broadcast_mul_relu, ref_broadcast_mul_relu, inputs, bench_torch=True
    )


# ---------------------------------------------------------------------------
# 04-2: Backward — dA[i, j] = dC[i, j] * B[j] * (A[i, j] * B[j] > 0)
# ---------------------------------------------------------------------------
r"""
Compute the gradient of the loss w.r.t. A through the broadcast-mul + ReLU operation.

Forward: C[i, j] = max(0, A[i, j] * B[j])
Backward: dA[i, j] = dC[i, j] * B[j] * (A[i, j] * B[j] > 0 ? 1 : 0)

Inputs:
    A:  Tensor([M, N], float32)
    B:  Tensor([N,], float32)
    dC: Tensor([M, N], float32)

Output:
    dA: Tensor([M, N], float32)
"""


def ref_backward_da(
    A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor
) -> torch.Tensor:
    # Use autograd to compute the reference gradient
    A_ag = A.clone().requires_grad_(True)
    C = torch.relu(A_ag * B[None, :])
    C.backward(dC)
    return A_ag.grad


@ct.kernel
def ct_backward_da(a, b, dc, da, TILE_M: ConstInt, TILE_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    # Create index tiles
    row_indices = bid_m * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    col_indices = bid_n * TILE_N + ct.arange(TILE_N, dtype=ct.int32)

    # Create 2D broadcast indices
    row_2d = row_indices[:, None]
    col_2d = col_indices[None, :]

    # Gather all needed tiles
    a_tile = ct.gather(a, (row_2d, col_2d))    # shape (TILE_M, TILE_N)
    dc_tile = ct.gather(dc, (row_2d, col_2d))  # shape (TILE_M, TILE_N)
    b_tile = ct.gather(b, col_indices)          # shape (TILE_N,)
    b_broadcast = b_tile[None, :]               # shape (1, TILE_N)

    # Recompute pre-ReLU activation
    pre_relu = a_tile * b_broadcast

    # ReLU gradient mask: 1 where pre_relu > 0, else 0
    mask = ct.greater(pre_relu, 0)

    # Compute dA: chain rule through ReLU and multiplication
    da_tile = ct.where(mask, dc_tile * b_broadcast, 0.0)

    # Scatter result to dA
    ct.scatter(da, (row_2d, col_2d), da_tile)


def launch_backward_da(
    A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor
) -> torch.Tensor:
    M, N = A.shape
    TILE_M, TILE_N = 32, 32
    dA = torch.empty_like(A)
    grid = (math.ceil(M / TILE_M), math.ceil(N / TILE_N), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        ct_backward_da,
        (A, B, dC, dA, TILE_M, TILE_N),
    )
    return dA


def run_backward_da():
    print("\n=== 04-2: Backward dL/dA (gather/scatter) ===\n")
    M, N = 256, 512
    inputs = {
        "A": torch.randn(M, N, dtype=torch.float32, device="cuda"),
        "B": torch.randn(N, dtype=torch.float32, device="cuda"),
        "dC": torch.randn(M, N, dtype=torch.float32, device="cuda"),
    }
    test_puzzle(
        launch_backward_da,
        ref_backward_da,
        inputs,
        label="04-2 Backward dL/dA (gather/scatter)",
    )
    bench_puzzle(launch_backward_da, ref_backward_da, inputs, bench_torch=True)


if __name__ == "__main__":
    run_broadcast_mul_relu()
    run_backward_da()
