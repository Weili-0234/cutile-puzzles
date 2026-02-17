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
Use one block per row tile. Load with zero padding so out-of-bounds values
do not affect the sum, accumulate in float32 for precision, then store one
value per row.

Inputs:
    A: Tensor([M, N], float32)

Output:
    B: Tensor([M,], float32)  where B[i] = sum(A[i, :])

HINT: Think in row tiles: load once, reduce across columns, and write one
output value per row.
"""


def ref_row_sum(A: torch.Tensor) -> torch.Tensor:
    return A.float().sum(dim=1)


@ct.kernel
def ct_row_sum(a, b, TILE_N: ConstInt):
    # Each block processes one row
    bid = ct.bid(0)

    # TODO: Implement single-tile row sum
    # Use one load/store path per row tile.
    # Accumulate in float32, then write the reduced value to b.
    # Ensure out-of-bounds elements are neutral for summation.
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
        hint="Use a row-tile reduction and keep accumulation in float32.",
    )


# ---------------------------------------------------------------------------
# 05-2: Multi-tile (Chunked) Row Sum
# ---------------------------------------------------------------------------
r"""
Compute the sum of each row when the row is too wide for a single tile:
B[i] = sum(A[i, :])

When N > TILE_N, we must loop over chunks of the row, accumulating
partial sums.
Use a running accumulator per row and sweep over row chunks. Each chunk
contributes a partial sum; combine all partial sums and write one final value.

NUM_CHUNKS = ceil(N / TILE_N), passed as a compile-time constant.

Inputs:
    A: Tensor([M, N], float32)

Output:
    B: Tensor([M,], float32)  where B[i] = sum(A[i, :])

HINT: Treat this as chunked reduction with a running accumulator per row.
"""


@ct.kernel
def ct_row_sum_chunked(a, b, TILE_N: ConstInt, NUM_CHUNKS: ConstInt):
    bid = ct.bid(0)

    # TODO: Implement chunked row sum
    # Iterate over row chunks and accumulate partial reductions in float32.
    # Keep the accumulator per row and store it after all chunks are processed.
    # Out-of-bounds values should not contribute to the result.
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
        hint="Use chunked row reduction with a running float32 accumulator.",
    )


if __name__ == "__main__":
    run_row_sum()
    run_row_sum_chunked()
