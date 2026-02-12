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

Inputs:
    A: Tensor([M, N], float32)

Output:
    B: Tensor([M,], float32)  where B[i] = sum(A[i, :])
"""


def ref_row_sum(A: torch.Tensor) -> torch.Tensor:
    return A.float().sum(dim=1)


@ct.kernel
def ct_row_sum(a, b, TILE_N: ConstInt):
    # Each block processes one row
    bid = ct.bid(0)

    # Load one row as a 2D tile of shape (1, TILE_N)
    tile = ct.load(a, index=(bid, 0), shape=(1, TILE_N), padding_mode=ct.PaddingMode.ZERO)

    # Cast to float32 for precision
    tile = ct.astype(tile, ct.float32)

    # Sum along columns (axis=1) to get a (1,) tile
    row_sum = ct.sum(tile, axis=1)

    # Store scalar result to b
    ct.store(b, index=(bid,), tile=row_sum.reshape((1,)))


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
    test_puzzle(launch_row_sum, ref_row_sum, inputs, label="05-1 Row Sum (single-tile)")
    bench_puzzle(launch_row_sum, ref_row_sum, inputs, bench_torch=True)


# ---------------------------------------------------------------------------
# 05-2: Multi-tile (Chunked) Row Sum
# ---------------------------------------------------------------------------
r"""
Compute the sum of each row when the row is too wide for a single tile:
B[i] = sum(A[i, :])

When N > TILE_N, we must loop over chunks of the row, accumulating
partial sums.

Inputs:
    A: Tensor([M, N], float32)

Output:
    B: Tensor([M,], float32)  where B[i] = sum(A[i, :])
"""


@ct.kernel
def ct_row_sum_chunked(a, b, TILE_N: ConstInt, NUM_CHUNKS: ConstInt):
    bid = ct.bid(0)

    # Initialize accumulator to zero
    acc = ct.full((1,), 0.0, dtype=ct.float32)

    # Loop over chunks of the row
    for chunk_idx in range(NUM_CHUNKS):
        # Load a (1, TILE_N) chunk of the row
        tile = ct.load(
            a, index=(bid, chunk_idx), shape=(1, TILE_N),
            padding_mode=ct.PaddingMode.ZERO,
        )

        # Cast to float32
        tile = ct.astype(tile, ct.float32)

        # Sum this chunk along columns
        partial_sum = ct.sum(tile, axis=1)

        # Accumulate
        acc = acc + partial_sum.reshape((1,))

    # Store final sum
    ct.store(b, index=(bid,), tile=acc)


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
        launch_row_sum_chunked, ref_row_sum, inputs,
        label="05-2 Row Sum (chunked)",
    )
    bench_puzzle(launch_row_sum_chunked, ref_row_sum, inputs, bench_torch=True)


if __name__ == "__main__":
    run_row_sum()
    run_row_sum_chunked()
