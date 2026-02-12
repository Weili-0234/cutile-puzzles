"""
Puzzle 05: Reduce Sum
======================
Learn how to perform reductions over rows of a matrix using cutile.
We explore both single-tile reductions and chunked (multi-tile) reductions
for rows that are too wide to fit in a single tile.

Category: ["official"]
Difficulty: ["medium"]
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
# 05-1: Single-tile Row Sum
# ---------------------------------------------------------------------------
r"""
Compute the sum of each row: B[i] = sum(A[i, :])

Assumes the entire row fits in one tile (N <= TILE_N), so each block
processes one row with a single load and reduce.

Algorithm:
    1. Use ct.bid(0) to get the row index (one block per row).
    2. Load the row as a 2D tile of shape (1, TILE_N) from A.
       Use padding_mode=ct.PaddingMode.ZERO so out-of-bounds elements
       contribute 0 to the sum.
    3. Upcast to float32 for numerical precision: ct.astype(tile, ct.float32).
    4. Reduce along axis=1: row_sum = ct.sum(tile, axis=1).
       This produces a tile of shape (1,).
    5. Store the scalar result to B at index (bid,) using the reshaped tile.

Inputs:
    A: Tensor([M, N], float32)

Output:
    B: Tensor([M,], float32)  where B[i] = sum(A[i, :])

HINT: ct.sum(tile, axis=1) reduces along columns. Use ct.astype to upcast
to float32 before summing. ct.load with padding_mode=ct.PaddingMode.ZERO
ensures OOB elements are zero-padded.
"""


def ref_row_sum(A: torch.Tensor) -> torch.Tensor:
    return A.float().sum(dim=1)


@ct.kernel
def ct_row_sum(a, b, TILE_N: ConstInt):
    # Each block processes one row
    bid = ct.bid(0)

    # TODO: Implement single-tile row sum
    # 1. Load a 2D tile of shape (1, TILE_N) from a at index (bid, 0)
    #    with padding_mode=ct.PaddingMode.ZERO
    # 2. Cast to float32 using ct.astype(tile, ct.float32)
    # 3. Sum along axis=1: ct.sum(tile, axis=1) -> shape (1,)
    # 4. Reshape to (1,) and store to b at index (bid,)
    pass


def launch_row_sum(A: torch.Tensor) -> torch.Tensor:
    M, N = A.shape
    TILE_N = 512
    B = torch.empty(M, dtype=torch.float32, device=A.device)
    grid = (M, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, ct_row_sum, (A, B, TILE_N))
    return B


def run_row_sum():
    print("\n=== 05-1: Single-tile Row Sum ===\n")
    M, N = 256, 512
    inputs = {
        "A": torch.randn(M, N, dtype=torch.float32, device="cuda"),
    }
    test_puzzle(
        launch_row_sum,
        ref_row_sum,
        inputs,
        label="05-1 Row Sum (single-tile)",
        hint="Load a (1, TILE_N) tile per row, ct.sum(tile, axis=1) to reduce, "
        "then store the scalar result.",
    )


# ---------------------------------------------------------------------------
# 05-2: Multi-tile (Chunked) Row Sum
# ---------------------------------------------------------------------------
r"""
Compute the sum of each row when the row is too wide for a single tile:
B[i] = sum(A[i, :])

When N > TILE_N, we must loop over chunks of the row, accumulating
partial sums.

Algorithm:
    1. Use ct.bid(0) to get the row index.
    2. Initialize an accumulator: acc = ct.full((1,), 0.0, dtype=ct.float32).
    3. Loop over chunks: for chunk_idx in range(NUM_CHUNKS):
        a. Load a (1, TILE_N) tile from A at index (bid, chunk_idx).
        b. Cast to float32.
        c. Sum along axis=1 to get a partial sum of shape (1,).
        d. Add to accumulator: acc = acc + partial_sum.
    4. Store the final accumulator to B.

NUM_CHUNKS = ceil(N / TILE_N), passed as a compile-time constant.

Inputs:
    A: Tensor([M, N], float32)

Output:
    B: Tensor([M,], float32)  where B[i] = sum(A[i, :])

HINT: Use ct.full((1,), 0.0, dtype=ct.float32) to create a zero accumulator.
Loop over chunks with a Python for-loop. ct.load handles out-of-bounds
padding automatically.
"""


@ct.kernel
def ct_row_sum_chunked(a, b, TILE_N: ConstInt, NUM_CHUNKS: ConstInt):
    bid = ct.bid(0)

    # TODO: Implement chunked row sum
    # 1. Initialize accumulator: ct.full((1,), 0.0, dtype=ct.float32)
    # 2. Loop over chunks: for chunk_idx in range(NUM_CHUNKS):
    #    a. Load (1, TILE_N) tile from a at index (bid, chunk_idx)
    #       with padding_mode=ct.PaddingMode.ZERO
    #    b. Cast to float32
    #    c. Compute partial sum: ct.sum(tile, axis=1)
    #    d. Add partial sum to accumulator
    # 3. Store accumulator to b at index (bid,)
    pass


def launch_row_sum_chunked(A: torch.Tensor) -> torch.Tensor:
    M, N = A.shape
    TILE_N = 512
    NUM_CHUNKS = math.ceil(N / TILE_N)
    B = torch.empty(M, dtype=torch.float32, device=A.device)
    grid = (M, 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        ct_row_sum_chunked,
        (A, B, TILE_N, NUM_CHUNKS),
    )
    return B


def run_row_sum_chunked():
    print("\n=== 05-2: Chunked Row Sum ===\n")
    M, N = 256, 4096
    inputs = {
        "A": torch.randn(M, N, dtype=torch.float32, device="cuda"),
    }
    test_puzzle(
        launch_row_sum_chunked,
        ref_row_sum,
        inputs,
        label="05-2 Row Sum (chunked)",
        hint="Initialize acc = ct.full((1,), 0.0, dtype=ct.float32). "
        "Loop over chunks, summing each (1, TILE_N) tile and adding to acc.",
    )


if __name__ == "__main__":
    run_row_sum()
    run_row_sum_chunked()
