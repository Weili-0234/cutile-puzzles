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

Inputs:
    A: Tensor([M, N], float16)

Output:
    B: Tensor([M, N], float16)  where B[i,:] = softmax(A[i,:])
"""


def ref_softmax(A: torch.Tensor) -> torch.Tensor:
    return torch.softmax(A.float(), dim=1).to(A.dtype)


@ct.kernel
def ct_softmax(a, b, TILE_N: ConstInt):
    bid = ct.bid(0)

    # Load row with NEG_INF padding (so OOB elements vanish after exp)
    tile = ct.load(
        a, index=(bid, 0), shape=(1, TILE_N),
        padding_mode=ct.PaddingMode.NEG_INF,
    )

    # Cast to float32 for numerical precision
    tile = ct.astype(tile, ct.float32)

    # Find row max for numerical stability
    row_max = ct.max(tile, axis=1, keepdims=True)

    # Subtract max
    shifted = tile - row_max

    # Exponentiate
    numerator = ct.exp(shifted)

    # Sum of exponentials
    denominator = ct.sum(numerator, axis=1, keepdims=True)

    # Divide to get softmax probabilities
    result = ct.truediv(numerator, denominator)

    # Cast back to original dtype and store
    result = ct.astype(result, ct.float16)
    ct.store(b, index=(bid, 0), tile=result)


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
    test_puzzle(launch_softmax, ref_softmax, inputs, label="06-1 Softmax (single-tile)")
    bench_puzzle(launch_softmax, ref_softmax, inputs, bench_torch=True)


# ---------------------------------------------------------------------------
# 06-2: Chunked Softmax (3-pass)
# ---------------------------------------------------------------------------
r"""
Compute row-wise softmax when the row doesn't fit in one tile.

Uses a 3-pass algorithm:
    Pass 1: Find row max across all chunks.
    Pass 2: Compute sum of exp(x - max) across all chunks.
    Pass 3: Compute exp(x - max) / sum for each chunk and store.

Inputs:
    A: Tensor([M, N], float16)

Output:
    B: Tensor([M, N], float16)  where B[i,:] = softmax(A[i,:])
"""


@ct.kernel
def ct_softmax_chunked(a, b, TILE_N: ConstInt, NUM_CHUNKS: ConstInt):
    bid = ct.bid(0)

    # Pass 1: Find the row maximum
    row_max = ct.full((1, 1), float("-inf"), dtype=ct.float32)
    for chunk_idx in range(NUM_CHUNKS):
        tile = ct.load(
            a, index=(bid, chunk_idx), shape=(1, TILE_N),
            padding_mode=ct.PaddingMode.NEG_INF,
        )
        tile = ct.astype(tile, ct.float32)
        chunk_max = ct.max(tile, axis=1, keepdims=True)
        row_max = ct.maximum(row_max, chunk_max)

    # Pass 2: Compute sum of exp(x - max)
    sum_exp = ct.full((1, 1), 0.0, dtype=ct.float32)
    for chunk_idx in range(NUM_CHUNKS):
        tile = ct.load(
            a, index=(bid, chunk_idx), shape=(1, TILE_N),
            padding_mode=ct.PaddingMode.NEG_INF,
        )
        tile = ct.astype(tile, ct.float32)
        numerator = ct.exp(tile - row_max)
        sum_exp = sum_exp + ct.sum(numerator, axis=1, keepdims=True)

    # Pass 3: Compute softmax and store
    for chunk_idx in range(NUM_CHUNKS):
        tile = ct.load(
            a, index=(bid, chunk_idx), shape=(1, TILE_N),
            padding_mode=ct.PaddingMode.NEG_INF,
        )
        tile = ct.astype(tile, ct.float32)
        result = ct.truediv(ct.exp(tile - row_max), sum_exp)
        result = ct.astype(result, ct.float16)
        ct.store(b, index=(bid, chunk_idx), tile=result)


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
        launch_softmax_chunked, ref_softmax, inputs,
        label="06-2 Softmax (chunked, 3-pass)",
    )
    bench_puzzle(launch_softmax_chunked, ref_softmax, inputs, bench_torch=True)


if __name__ == "__main__":
    run_softmax()
    run_softmax_chunked()
