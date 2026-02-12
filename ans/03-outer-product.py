"""
Puzzle 03: Outer Product
=========================
Learn to produce 2D outputs from 1D inputs using broadcasting.
Explore both load/store with reshape and gather/scatter with 2D index broadcasting.

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
# 03-1: Outer Add (load/store with reshape) — C[i, j] = A[i] + B[j]
# ---------------------------------------------------------------------------
r"""
Compute the outer addition of two 1D vectors, producing a 2D matrix.

C[i, j] = A[i] + B[j]

Strategy: Load 1D tiles, reshape to enable broadcasting, add, store 2D result.

Inputs:
    A: Tensor([M,], float32)
    B: Tensor([N,], float32)

Output:
    C: Tensor([M, N], float32)  where C[i, j] = A[i] + B[j]
"""


def ref_outer_add(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A[:, None] + B[None, :]


@ct.kernel
def ct_outer_add_loadstore(a, b, c, TILE_M: ConstInt, TILE_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    # Load 1D tiles from A and B
    a_tile = ct.load(a, index=(bid_m,), shape=(TILE_M,))
    b_tile = ct.load(b, index=(bid_n,), shape=(TILE_N,))

    # Reshape for broadcasting: (TILE_M, 1) + (1, TILE_N) -> (TILE_M, TILE_N)
    a_2d = a_tile.reshape((TILE_M, 1))
    b_2d = b_tile.reshape((1, TILE_N))

    # Outer addition via broadcasting
    c_tile = a_2d + b_2d

    # Store the 2D result
    ct.store(c, index=(bid_m, bid_n), tile=c_tile)


def launch_outer_add_loadstore(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M = A.shape[0]
    N = B.shape[0]
    TILE_M, TILE_N = 32, 32
    C = torch.empty(M, N, dtype=torch.float32, device=A.device)
    grid = (math.ceil(M / TILE_M), math.ceil(N / TILE_N), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        ct_outer_add_loadstore,
        (A, B, C, TILE_M, TILE_N),
    )
    return C


def run_outer_add_loadstore():
    print("\n=== 03-1: Outer Add (load/store with reshape) ===\n")
    M, N = 256, 512
    inputs = {
        "A": torch.randn(M, dtype=torch.float32, device="cuda"),
        "B": torch.randn(N, dtype=torch.float32, device="cuda"),
    }
    test_puzzle(
        launch_outer_add_loadstore,
        ref_outer_add,
        inputs,
        label="03-1 Outer Add (load/store)",
    )
    bench_puzzle(launch_outer_add_loadstore, ref_outer_add, inputs, bench_torch=True)


# ---------------------------------------------------------------------------
# 03-2: Outer Add (gather/scatter with broadcasting)
# ---------------------------------------------------------------------------
r"""
Same operation C[i, j] = A[i] + B[j], but using gather/scatter with index broadcasting.

Inputs:
    A: Tensor([M,], float32)
    B: Tensor([N,], float32)

Output:
    C: Tensor([M, N], float32)  where C[i, j] = A[i] + B[j]
"""


@ct.kernel
def ct_outer_add_gather(a, b, c, TILE_M: ConstInt, TILE_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    # Compute 1D indices for this tile
    row_indices = bid_m * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    col_indices = bid_n * TILE_N + ct.arange(TILE_N, dtype=ct.int32)

    # Gather 1D values from A and B
    a_tile = ct.gather(a, row_indices)   # shape (TILE_M,)
    b_tile = ct.gather(b, col_indices)   # shape (TILE_N,)

    # Broadcast-add to get 2D result
    result = a_tile[:, None] + b_tile[None, :]  # shape (TILE_M, TILE_N)

    # Create 2D scatter indices using broadcasting
    row_idx_2d = row_indices[:, None]  # shape (TILE_M, 1)
    col_idx_2d = col_indices[None, :]  # shape (1, TILE_N)

    # Scatter the 2D result into C
    ct.scatter(c, (row_idx_2d, col_idx_2d), result)


def launch_outer_add_gather(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M = A.shape[0]
    N = B.shape[0]
    TILE_M, TILE_N = 32, 32
    C = torch.empty(M, N, dtype=torch.float32, device=A.device)
    grid = (math.ceil(M / TILE_M), math.ceil(N / TILE_N), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        ct_outer_add_gather,
        (A, B, C, TILE_M, TILE_N),
    )
    return C


def run_outer_add_gather():
    print("\n=== 03-2: Outer Add (gather/scatter with broadcasting) ===\n")
    M, N = 256, 512
    inputs = {
        "A": torch.randn(M, dtype=torch.float32, device="cuda"),
        "B": torch.randn(N, dtype=torch.float32, device="cuda"),
    }
    test_puzzle(
        launch_outer_add_gather,
        ref_outer_add,
        inputs,
        label="03-2 Outer Add (gather/scatter)",
    )
    bench_puzzle(launch_outer_add_gather, ref_outer_add, inputs, bench_torch=True)


if __name__ == "__main__":
    run_outer_add_loadstore()
    run_outer_add_gather()
