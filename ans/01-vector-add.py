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
"""


def ref_vec_add(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A + B


@ct.kernel
def ct_vec_add_1d(a, b, c, TILE: ConstInt):
    # Get the block index — each block processes one tile
    bid = ct.bid(0)

    # Load tiles from arrays a and b
    a_tile = ct.load(a, index=(bid,), shape=(TILE,))
    b_tile = ct.load(b, index=(bid,), shape=(TILE,))

    # Element-wise addition
    c_tile = a_tile + b_tile

    # Store the result
    ct.store(c, index=(bid,), tile=c_tile)


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
    test_puzzle(launch_vec_add_1d, ref_vec_add, inputs, label="01-1 Vec Add (load/store)")
    bench_puzzle(launch_vec_add_1d, ref_vec_add, inputs, bench_torch=True)


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
"""


@ct.kernel
def ct_vec_add_1d_gather(a, b, c, TILE: ConstInt):
    bid = ct.bid(0)

    # Compute global indices for this block's tile
    indices = bid * TILE + ct.arange(TILE, dtype=ct.int32)

    # Gather elements at those indices (out-of-bounds → 0)
    a_tile = ct.gather(a, indices)
    b_tile = ct.gather(b, indices)

    # Element-wise addition
    c_tile = a_tile + b_tile

    # Scatter results back (out-of-bounds writes are ignored)
    ct.scatter(c, indices, c_tile)


def launch_vec_add_1d_gather(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    N = A.shape[0]
    TILE = 1024
    C = torch.empty_like(A)
    grid = (math.ceil(N / TILE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, ct_vec_add_1d_gather, (A, B, C, TILE))
    return C


def run_vec_add_1d_gather():
    print("\n=== 01-2: Vector Add (gather/scatter) ===\n")
    # Use non-power-of-2 size to show boundary handling
    N = 4000
    inputs = {
        "A": torch.randn(N, dtype=torch.float32, device="cuda"),
        "B": torch.randn(N, dtype=torch.float32, device="cuda"),
    }
    test_puzzle(
        launch_vec_add_1d_gather,
        ref_vec_add,
        inputs,
        label="01-2 Vec Add (gather/scatter)",
        hint="Check your index computation: indices = bid * TILE + ct.arange(TILE, ...)",
    )
    bench_puzzle(launch_vec_add_1d_gather, ref_vec_add, inputs, bench_torch=True)


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
"""


def ref_mat_add(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A + B


@ct.kernel
def ct_mat_add_2d(a, b, c, TILE_M: ConstInt, TILE_N: ConstInt):
    # 2D block indexing
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    # Load 2D tiles
    a_tile = ct.load(a, index=(bid_m, bid_n), shape=(TILE_M, TILE_N))
    b_tile = ct.load(b, index=(bid_m, bid_n), shape=(TILE_M, TILE_N))

    # Element-wise addition
    c_tile = a_tile + b_tile

    # Store result
    ct.store(c, index=(bid_m, bid_n), tile=c_tile)


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
        hint="Make sure you use ct.bid(0) for rows and ct.bid(1) for columns.",
    )
    bench_puzzle(launch_mat_add_2d, ref_mat_add, inputs, bench_torch=True)


if __name__ == "__main__":
    run_vec_add_1d()
    run_vec_add_1d_gather()
    run_mat_add_2d()
