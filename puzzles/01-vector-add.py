"""
Puzzle 01: Vector Addition
==========================
Learn the fundamentals of cutile: loading data, computing, and storing results.
We explore both memory access patterns: load/store (structured) and gather/scatter (indexed).

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
# 01-1: 1D Vector Addition using ct.load / ct.store
# ---------------------------------------------------------------------------
r"""
The most basic cutile pattern: partition arrays into tiles and process one tile per block.

`ct.load(array, index=(i,), shape=(TILE,))` loads the i-th tile of size TILE from the array.
`ct.store(array, index=(i,), tile=result)` stores the tile back.
Out-of-bounds elements are automatically zero-padded on load and ignored on store.

Inputs:
    A: Tensor([N,], float32)
    B: Tensor([N,], float32)

Output:
    C: Tensor([N,], float32)  where C[i] = A[i] + B[i]

HINT: Use ct.bid(0) to get the block index. Each block processes one tile.
"""


def ref_vec_add(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A + B


@ct.kernel
def ct_vec_add_1d(a, b, c, TILE: ConstInt):
    # Get the block index — each block processes one tile
    bid = ct.bid(0)

    # TODO: Implement 1D vector addition
    # 1. Load tiles of size TILE from arrays a and b at index (bid,)
    # 2. Add them element-wise
    # 3. Store the result to array c at index (bid,)
    pass


def launch_vec_add_1d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    N = A.shape[0]
    TILE = 1024
    C = torch.empty_like(A)
    grid = (math.ceil(N / TILE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, ct_vec_add_1d, (A, B, C, TILE))
    return C


def run_vec_add_1d():
    print("\n=== 01-1: Vector Add (load/store) ===\n")
    N = 4096
    inputs = {
        "A": torch.randn(N, dtype=torch.float32, device="cuda"),
        "B": torch.randn(N, dtype=torch.float32, device="cuda"),
    }
    test_puzzle(
        launch_vec_add_1d,
        ref_vec_add,
        inputs,
        label="01-1 Vec Add (load/store)",
        hint="Use ct.load(a, index=(bid,), shape=(TILE,)) to load a tile, then ct.store to write.",
    )


# ---------------------------------------------------------------------------
# 01-2: 1D Vector Addition using ct.gather / ct.scatter
# ---------------------------------------------------------------------------
r"""
An alternative memory access pattern using explicit indices.

`ct.arange(TILE, dtype=ct.int32)` creates a tile [0, 1, ..., TILE-1].
`ct.gather(array, indices)` loads elements at the given indices (out-of-bounds → 0).
`ct.scatter(array, indices, tile)` stores elements at the given indices (out-of-bounds ignored).

This pattern is useful when indices are non-contiguous or computed dynamically.

Inputs:
    A: Tensor([N,], float32)
    B: Tensor([N,], float32)

Output:
    C: Tensor([N,], float32)  where C[i] = A[i] + B[i]

HINT: Compute global indices as `bid * TILE + ct.arange(TILE, dtype=ct.int32)`.
"""


@ct.kernel
def ct_vec_add_1d_gather(a, b, c, TILE: ConstInt):
    bid = ct.bid(0)

    # TODO: Implement 1D vector addition using gather/scatter
    # 1. Compute global indices: bid * TILE + ct.arange(TILE, dtype=ct.int32)
    # 2. Gather elements from a and b at those indices
    # 3. Add them element-wise
    # 4. Scatter the result to c at those indices
    pass


def launch_vec_add_1d_gather(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    N = A.shape[0]
    TILE = 1024
    C = torch.empty_like(A)
    grid = (math.ceil(N / TILE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, ct_vec_add_1d_gather, (A, B, C, TILE))
    return C


def run_vec_add_1d_gather():
    print("\n=== 01-2: Vector Add (gather/scatter) ===\n")
    N = 4000  # Non-power-of-2 to show boundary handling
    inputs = {
        "A": torch.randn(N, dtype=torch.float32, device="cuda"),
        "B": torch.randn(N, dtype=torch.float32, device="cuda"),
    }
    test_puzzle(
        launch_vec_add_1d_gather,
        ref_vec_add,
        inputs,
        label="01-2 Vec Add (gather/scatter)",
        hint="indices = bid * TILE + ct.arange(TILE, dtype=ct.int32). "
        "gather/scatter auto-handle out-of-bounds.",
    )


# ---------------------------------------------------------------------------
# 01-3: 2D Matrix Addition using ct.load / ct.store
# ---------------------------------------------------------------------------
r"""
Extend to 2D: use a 2D grid with ct.bid(0) and ct.bid(1) for row and column tile indices.

`ct.load(array, index=(bid_x, bid_y), shape=(TILE_M, TILE_N))` loads a 2D tile.
`ct.store(array, index=(bid_x, bid_y), tile=result)` stores a 2D tile.

Inputs:
    A: Tensor([M, N], float32)
    B: Tensor([M, N], float32)

Output:
    C: Tensor([M, N], float32)  where C[i, j] = A[i, j] + B[i, j]

HINT: Use ct.bid(0) for the row tile index and ct.bid(1) for the column tile index.
"""


def ref_mat_add(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A + B


@ct.kernel
def ct_mat_add_2d(a, b, c, TILE_M: ConstInt, TILE_N: ConstInt):
    # TODO: Implement 2D matrix addition
    # 1. Get bid_m = ct.bid(0) and bid_n = ct.bid(1)
    # 2. Load 2D tiles from a and b at index (bid_m, bid_n)
    # 3. Add them element-wise
    # 4. Store the result to c
    pass


def launch_mat_add_2d(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, N = A.shape
    TILE_M, TILE_N = 32, 32
    C = torch.empty_like(A)
    grid = (math.ceil(M / TILE_M), math.ceil(N / TILE_N), 1)
    ct.launch(torch.cuda.current_stream(), grid, ct_mat_add_2d, (A, B, C, TILE_M, TILE_N))
    return C


def run_mat_add_2d():
    print("\n=== 01-3: Matrix Add 2D (load/store) ===\n")
    M, N = 256, 512
    inputs = {
        "A": torch.randn(M, N, dtype=torch.float32, device="cuda"),
        "B": torch.randn(M, N, dtype=torch.float32, device="cuda"),
    }
    test_puzzle(
        launch_mat_add_2d,
        ref_mat_add,
        inputs,
        label="01-3 Mat Add 2D (load/store)",
        hint="Use ct.bid(0) for rows, ct.bid(1) for columns, and shape=(TILE_M, TILE_N).",
    )


if __name__ == "__main__":
    run_vec_add_1d()
    run_vec_add_1d_gather()
    run_mat_add_2d()
