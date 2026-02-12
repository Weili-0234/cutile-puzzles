"""
Puzzle 06: Softmax
===================
Learn how to implement the numerically-stable softmax function in cutile.
We explore both single-tile softmax (entire row fits in one tile) and
chunked softmax (3-pass algorithm for large rows).

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
# 06-1: Single-tile Softmax
# ---------------------------------------------------------------------------
r"""
Compute row-wise softmax: B[i, :] = softmax(A[i, :])

Assumes the entire row fits in one tile (N <= TILE_N).
Uses the numerically stable algorithm: subtract the max before exponentiating.

Algorithm:
    1. Use ct.bid(0) to get the row index (one block per row).
    2. Load the row as a (1, TILE_N) tile from A.
       Use padding_mode=ct.PaddingMode.NEG_INF so out-of-bounds elements
       become -inf (which gives exp(-inf)=0, neutral for softmax).
    3. Cast to float32 for numerical precision.
    4. Find the row max: row_max = ct.max(tile, axis=1, keepdims=True).
    5. Subtract max for stability: shifted = tile - row_max.
    6. Exponentiate: numerator = ct.exp(shifted).
    7. Sum: denominator = ct.sum(numerator, axis=1, keepdims=True).
    8. Divide: result = ct.truediv(numerator, denominator).
    9. Cast back to original dtype and store.

Inputs:
    A: Tensor([M, N], float16)

Output:
    B: Tensor([M, N], float16)  where B[i,:] = softmax(A[i,:])

HINT: Use padding_mode=ct.PaddingMode.NEG_INF so OOB elements are -inf
(exp(-inf)=0). Subtract ct.max before ct.exp for numerical stability.
ct.truediv(a, b) for division.
"""


def ref_softmax(A: torch.Tensor) -> torch.Tensor:
    return torch.softmax(A.float(), dim=1).to(A.dtype)


@ct.kernel
def ct_softmax(a, b, TILE_N: ConstInt):
    bid = ct.bid(0)

    # TODO: Implement single-tile softmax
    # 1. Load (1, TILE_N) tile from a at (bid, 0)
    #    with padding_mode=ct.PaddingMode.NEG_INF
    # 2. Cast to float32: ct.astype(tile, ct.float32)
    # 3. Row max: ct.max(tile, axis=1, keepdims=True)
    # 4. Subtract max: shifted = tile - row_max
    # 5. Exponentiate: ct.exp(shifted)
    # 6. Sum: ct.sum(numerator, axis=1, keepdims=True)
    # 7. Divide: ct.truediv(numerator, denominator)
    # 8. Cast back: ct.astype(result, ct.float16)
    # 9. Store to b at (bid, 0)
    pass


def launch_softmax(A: torch.Tensor) -> torch.Tensor:
    M, N = A.shape
    TILE_N = 256
    B = torch.empty_like(A)
    grid = (M, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, ct_softmax, (A, B, TILE_N))
    return B


def run_softmax():
    print("\n=== 06-1: Single-tile Softmax ===\n")
    M, N = 256, 256
    inputs = {
        "A": torch.randn(M, N, dtype=torch.float16, device="cuda"),
    }
    test_puzzle(
        launch_softmax,
        ref_softmax,
        inputs,
        label="06-1 Softmax (single-tile)",
        hint="Use NEG_INF padding so OOB elements vanish after exp. "
        "Subtract ct.max for stability, then exp / sum.",
    )


# ---------------------------------------------------------------------------
# 06-2: Chunked Softmax (3-pass)
# ---------------------------------------------------------------------------
r"""
Compute row-wise softmax when the row doesn't fit in one tile.

This requires a 3-pass algorithm:
    Pass 1 — Find row max: iterate over chunks, track running max.
    Pass 2 — Compute sum of exp(x - max): iterate over chunks again.
    Pass 3 — Compute softmax(x) = exp(x - max) / sum: iterate and store.

Algorithm:
    Pass 1:
        row_max = ct.full((1, 1), float('-inf'), dtype=ct.float32)
        for chunk_idx in range(NUM_CHUNKS):
            tile = load chunk with NEG_INF padding, cast to float32
            chunk_max = ct.max(tile, axis=1, keepdims=True)
            row_max = ct.maximum(row_max, chunk_max)

    Pass 2:
        sum_exp = ct.full((1, 1), 0.0, dtype=ct.float32)
        for chunk_idx in range(NUM_CHUNKS):
            tile = load chunk with NEG_INF padding, cast to float32
            numerator = ct.exp(tile - row_max)
            sum_exp = sum_exp + ct.sum(numerator, axis=1, keepdims=True)

    Pass 3:
        for chunk_idx in range(NUM_CHUNKS):
            tile = load chunk with NEG_INF padding, cast to float32
            result = ct.truediv(ct.exp(tile - row_max), sum_exp)
            cast back and store

Inputs:
    A: Tensor([M, N], float16)

Output:
    B: Tensor([M, N], float16)  where B[i,:] = softmax(A[i,:])

HINT: Use ct.full to initialize row_max to -inf and sum_exp to 0.
ct.maximum(a, b) for element-wise max. 3 separate loops over chunks.
"""


@ct.kernel
def ct_softmax_chunked(a, b, TILE_N: ConstInt, NUM_CHUNKS: ConstInt):
    bid = ct.bid(0)

    # TODO: Implement chunked softmax (3-pass algorithm)
    #
    # Pass 1: Find row max
    # 1. Init: row_max = ct.full((1, 1), float('-inf'), dtype=ct.float32)
    # 2. Loop over chunks:
    #    a. Load (1, TILE_N) tile with NEG_INF padding
    #    b. Cast to float32
    #    c. chunk_max = ct.max(tile, axis=1, keepdims=True)
    #    d. row_max = ct.maximum(row_max, chunk_max)
    #
    # Pass 2: Compute sum of exp(x - max)
    # 1. Init: sum_exp = ct.full((1, 1), 0.0, dtype=ct.float32)
    # 2. Loop over chunks:
    #    a. Load and cast same as pass 1
    #    b. numerator = ct.exp(tile - row_max)
    #    c. sum_exp = sum_exp + ct.sum(numerator, axis=1, keepdims=True)
    #
    # Pass 3: Compute and store softmax
    # 1. Loop over chunks:
    #    a. Load and cast same as pass 1
    #    b. result = ct.truediv(ct.exp(tile - row_max), sum_exp)
    #    c. Cast back to float16 and store to b at (bid, chunk_idx)
    pass


def launch_softmax_chunked(A: torch.Tensor) -> torch.Tensor:
    M, N = A.shape
    TILE_N = 512
    NUM_CHUNKS = math.ceil(N / TILE_N)
    B = torch.empty_like(A)
    grid = (M, 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        ct_softmax_chunked,
        (A, B, TILE_N, NUM_CHUNKS),
    )
    return B


def run_softmax_chunked():
    print("\n=== 06-2: Chunked Softmax ===\n")
    M, N = 256, 4096
    inputs = {
        "A": torch.randn(M, N, dtype=torch.float16, device="cuda"),
    }
    test_puzzle(
        launch_softmax_chunked,
        ref_softmax,
        inputs,
        label="06-2 Softmax (chunked, 3-pass)",
        hint="Three passes: (1) find row max with ct.maximum, "
        "(2) sum exp(x-max), (3) compute exp(x-max)/sum and store.",
    )


if __name__ == "__main__":
    run_softmax()
    run_softmax_chunked()
