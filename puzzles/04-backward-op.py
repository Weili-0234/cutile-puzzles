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

# Allow running from any directory
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

Strategy using gather/scatter:
  1. Use a 2D grid: ct.bid(0) for rows, ct.bid(1) for columns.
  2. Create 2D indices for A and C, and 1D indices for B.
  3. Gather A tile (2D) and B tile (1D), broadcast B across rows.
  4. Multiply, then apply ReLU using ct.where and ct.greater.
  5. Scatter the result to C.

Inputs:
    A: Tensor([M, N], float32)
    B: Tensor([N,], float32)

Output:
    C: Tensor([M, N], float32)  where C[i, j] = max(0, A[i, j] * B[j])

HINT: Gather B with col_indices (1D), then use b_tile[None, :] to broadcast
      across rows before multiplying with the 2D A tile.
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

    # TODO: Implement broadcast multiply + ReLU using gather/scatter
    # 1. Gather 2D tile from A using (row_indices[:, None], col_indices[None, :])
    # 2. Gather 1D tile from B using col_indices
    # 3. Broadcast B: b_tile[None, :]  -> shape (1, TILE_N)
    # 4. Multiply: product = a_tile * b_broadcast
    # 5. ReLU: mask = ct.greater(product, 0); result = ct.where(mask, product, 0.0)
    # 6. Scatter result to C at (row_indices[:, None], col_indices[None, :])
    pass


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
        hint="Gather B as 1D, then b_tile[None, :] broadcasts it to (1, TILE_N). "
        "Multiply with 2D A tile, then apply ct.where(ct.greater(product, 0), product, 0.0).",
    )


# ---------------------------------------------------------------------------
# 04-2: Backward — dA[i, j] = dC[i, j] * B[j] * (A[i, j] * B[j] > 0)
# ---------------------------------------------------------------------------
r"""
Compute the gradient of the loss w.r.t. A through the broadcast-mul + ReLU operation.

The forward operation was: C[i, j] = max(0, A[i, j] * B[j])

By the chain rule through ReLU:
  dA[i, j] = dC[i, j] * B[j]      if A[i, j] * B[j] > 0
  dA[i, j] = 0                     otherwise

Equivalently: dA[i, j] = dC[i, j] * B[j] * (A[i, j] * B[j] > 0 ? 1 : 0)

This is because:
  - d/dA[i,j] (A[i,j] * B[j]) = B[j]  (partial derivative of multiplication)
  - d/dx (ReLU(x)) = 1 if x > 0, else 0  (ReLU gradient)
  - Chain rule: dA = dC * B[j] * relu_grad

Inputs:
    A:  Tensor([M, N], float32) — original input
    B:  Tensor([N,], float32)   — original broadcast vector
    dC: Tensor([M, N], float32) — upstream gradient

Output:
    dA: Tensor([M, N], float32)

HINT: Recompute pre_relu = A[i,j] * B[j]. Use ct.greater(pre_relu, 0) for the mask.
      Then dA = ct.where(mask, dC_tile * b_broadcast, 0.0).
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

    # TODO: Implement backward pass for dL/dA
    # 1. Gather A tile (2D) from a at (row_indices[:, None], col_indices[None, :])
    # 2. Gather B tile (1D) from b at col_indices, broadcast: b_tile[None, :]
    # 3. Gather dC tile (2D) from dc at (row_indices[:, None], col_indices[None, :])
    # 4. Recompute pre-ReLU: pre_relu = a_tile * b_broadcast
    # 5. ReLU mask: mask = ct.greater(pre_relu, 0)
    # 6. Gradient: da_tile = ct.where(mask, dc_tile * b_broadcast, 0.0)
    # 7. Scatter da_tile to da at (row_indices[:, None], col_indices[None, :])
    pass


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
        hint="Recompute pre_relu = a_tile * b_broadcast. "
        "mask = ct.greater(pre_relu, 0). "
        "da_tile = ct.where(mask, dc_tile * b_broadcast, 0.0).",
    )


if __name__ == "__main__":
    run_broadcast_mul_relu()
    run_backward_da()
