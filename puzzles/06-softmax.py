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
Use one block per row tile. Load with NEG_INF padding so out-of-bounds
elements vanish after exponentiation, perform stable softmax in float32,
then cast back and store.

Inputs:
    A: Tensor([M, N], float16)

Output:
    B: Tensor([M, N], float16)  where B[i,:] = softmax(A[i,:])

HINT: Keep softmax numerically stable: max-shift before exponentiation, and
use padding that does not contribute probability mass.
"""


def ref_softmax(A: torch.Tensor) -> torch.Tensor:
    return torch.softmax(A.float(), dim=1).to(A.dtype)


@ct.kernel
def ct_softmax(a, b, TILE_N: ConstInt):
    bid = ct.bid(0)

    # TODO: Implement single-tile softmax
    # Compute stable row-wise softmax for one tile.
    # Use float32 intermediates for numerical robustness.
    # Ensure out-of-bounds elements do not affect normalization.
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
        hint="Use a stable max-shifted softmax and neutral OOB padding.",
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
Each pass has a different role: collect a stable reference max, accumulate
the normalization factor, then emit normalized outputs.

Inputs:
    A: Tensor([M, N], float16)

Output:
    B: Tensor([M, N], float16)  where B[i,:] = softmax(A[i,:])

HINT: Keep the three passes conceptually separate: max pass, sum-exp pass,
then output pass.
"""


@ct.kernel
def ct_softmax_chunked(a, b, TILE_N: ConstInt, NUM_CHUNKS: ConstInt):
    bid = ct.bid(0)

    # TODO: Implement chunked softmax (3-pass algorithm)
    # Implement the three-pass stable softmax over chunks:
    # track row max, accumulate normalization, then write normalized chunks.
    # Use float32 intermediates and padding that is neutral for softmax.
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
        hint="Use a stable three-pass chunked softmax: max, normalize factor, output.",
    )


if __name__ == "__main__":
    run_softmax()
    run_softmax_chunked()
