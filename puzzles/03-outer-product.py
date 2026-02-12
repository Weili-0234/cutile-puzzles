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

# Allow running from any directory
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

Strategy using load/store with reshape:
  1. Use a 2D grid: ct.bid(0) for rows, ct.bid(1) for columns.
  2. Load 1D tiles from A and B using ct.load with shape (TILE_M,) and (TILE_N,).
  3. Reshape to enable broadcasting:
     - a_tile.reshape((TILE_M, 1))  -> shape (TILE_M, 1)
     - b_tile.reshape((1, TILE_N))  -> shape (1, TILE_N)
  4. Add: broadcasting produces shape (TILE_M, TILE_N).
  5. Store the 2D result tile to C.

Inputs:
    A: Tensor([M,], float32)
    B: Tensor([N,], float32)

Output:
    C: Tensor([M, N], float32)  where C[i, j] = A[i] + B[j]

HINT: Use a_tile.reshape((TILE_M, 1)) + b_tile.reshape((1, TILE_N)) for broadcasting.
"""


def ref_outer_add(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A[:, None] + B[None, :]


@ct.kernel
def ct_outer_add_loadstore(a, b, c, TILE_M: ConstInt, TILE_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    # TODO: Implement outer add using load/store with reshape
    # 1. Load a 1D tile from A at index (bid_m,) with shape (TILE_M,)
    # 2. Load a 1D tile from B at index (bid_n,) with shape (TILE_N,)
    # 3. Reshape a_tile to (TILE_M, 1) and b_tile to (1, TILE_N)
    # 4. Add the reshaped tiles (broadcasting produces (TILE_M, TILE_N))
    # 5. Store the 2D result to C at index (bid_m, bid_n)
    pass


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
        hint="Load 1D tiles, reshape to (TILE_M,1) and (1,TILE_N), then add for broadcasting.",
    )


# ---------------------------------------------------------------------------
# 03-2: Outer Add (gather/scatter with broadcasting)
# ---------------------------------------------------------------------------
r"""
Same operation C[i, j] = A[i] + B[j], but using gather/scatter with index broadcasting.

Strategy using gather/scatter:
  1. Create 1D index tiles for rows and columns using ct.arange.
  2. Gather from A using row_indices (1D) -> a_tile shape (TILE_M,)
  3. Gather from B using col_indices (1D) -> b_tile shape (TILE_N,)
  4. Reshape/expand_dims to broadcast: a_tile[:, None] + b_tile[None, :]
  5. Create 2D index arrays for scatter:
     - row_idx_2d = row_indices[:, None]  shape (TILE_M, 1)
     - col_idx_2d = col_indices[None, :]  shape (1, TILE_N)
  6. Scatter result into C using (row_idx_2d, col_idx_2d) — broadcasting handles it.

Inputs:
    A: Tensor([M,], float32)
    B: Tensor([N,], float32)

Output:
    C: Tensor([M, N], float32)  where C[i, j] = A[i] + B[j]

HINT: Use expand_dims (e.g., a_tile[:, None]) to set up broadcasting for both
      the computation and the scatter indices.
"""


@ct.kernel
def ct_outer_add_gather(a, b, c, TILE_M: ConstInt, TILE_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    # TODO: Implement outer add using gather/scatter with broadcasting
    # 1. Compute row_indices = bid_m * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    # 2. Compute col_indices = bid_n * TILE_N + ct.arange(TILE_N, dtype=ct.int32)
    # 3. Gather 1D from A at row_indices, gather 1D from B at col_indices
    # 4. Broadcast-add: a_tile[:, None] + b_tile[None, :]
    # 5. Create 2D scatter indices: (row_indices[:, None], col_indices[None, :])
    # 6. Scatter result into C at those 2D indices
    pass


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
        hint="Use a_tile[:, None] + b_tile[None, :] for broadcasting, and "
        "(row_indices[:, None], col_indices[None, :]) for 2D scatter indices.",
    )


if __name__ == "__main__":
    run_outer_add_loadstore()
    run_outer_add_gather()
