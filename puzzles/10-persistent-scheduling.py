"""
Puzzle 10: Persistent Scheduling
=================================
Learn persistent kernel scheduling: instead of one block per work item,
launch fewer blocks that each loop over multiple work items with stride = num_blocks.
This improves GPU occupancy and reduces launch overhead.

Category: ["official"]
Difficulty: ["hard"]
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

NUM_SM = torch.cuda.get_device_properties("cuda").multi_processor_count


# ---------------------------------------------------------------------------
# 10-1: Persistent Softmax
# ---------------------------------------------------------------------------
r"""
Compute row-wise softmax with persistent scheduling.

Instead of launching one block per row (like puzzle 06), launch fewer blocks
that each process multiple rows in a loop. This is the "persistent kernel" pattern.

Each block starts at its own bid and strides by the total number of blocks,
ensuring all rows are covered without conflicts.

Uses gather/scatter for single-tile-per-row softmax with -inf padding for OOB.
Within each assigned row, perform a numerically stable softmax and write the
result back with boundary-safe indexing.

Inputs:
    A: Tensor([M, N], float16)

Output:
    B: Tensor([M, N], float16)  where B[i,:] = softmax(A[i,:])

HINT: Use a persistent row loop, and keep softmax numerically stable per row.
"""


def ref_persistent_softmax(A: torch.Tensor) -> torch.Tensor:
    return torch.softmax(A.float(), dim=1).to(A.dtype)


@ct.kernel(occupancy=4)
def ct_persistent_softmax(a, b, n_rows: ConstInt, TILE_N: ConstInt):
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)
    offsets = ct.arange(TILE_N, dtype=ct.int32)

    # TODO: Implement persistent softmax
    # Iterate persistently over assigned rows for this block.
    # For each row, run stable softmax in float32 and scatter back safely.
    # Keep out-of-bounds elements neutral for softmax.
    pass


def launch_persistent_softmax(A: torch.Tensor) -> torch.Tensor:
    M, N = A.shape
    TILE_N = 256
    B = torch.empty_like(A)
    occupancy = 4
    num_programs = min(NUM_SM * occupancy, M)
    grid = (num_programs, 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        ct_persistent_softmax,
        (A, B, M, TILE_N),
    )
    return B


def run_persistent_softmax():
    print("\n=== 10-1: Persistent Softmax ===\n")
    M, N = 4096, 256
    inputs = {
        "A": torch.randn(M, N, dtype=torch.float16, device="cuda"),
    }
    test_puzzle(
        launch_persistent_softmax,
        ref_persistent_softmax,
        inputs,
        label="10-1 Persistent Softmax",
        hint="Use persistent row scheduling and stable softmax for each assigned row.",
    )


# ---------------------------------------------------------------------------
# 10-2: Persistent GEMM
# ---------------------------------------------------------------------------
r"""
Matrix multiplication C = A @ B with persistent scheduling.

Instead of a 2D grid (one block per output tile), use a 1D grid where
each block processes multiple output tiles in a persistent loop.

The key idea is mapping 1D block IDs to 2D tile coordinates, then doing
a standard tiled GEMM (load A tile, load B tile, ct.mma, accumulate).

Inputs:
    A: Tensor([M, K], float16)
    B: Tensor([K, N], float16)

Output:
    C: Tensor([M, N], float16)  where C = A @ B

HINT: Use persistent tile scheduling, then run tiled GEMM with ct.mma per tile.
"""


def ref_persistent_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (A.float() @ B.float()).to(A.dtype)


@ct.kernel(occupancy=2)
def ct_persistent_gemm(
    a, b, c,
    M: ConstInt, N: ConstInt, K: ConstInt,
    TILE_M: ConstInt, TILE_N: ConstInt, TILE_K: ConstInt,
):
    bid = ct.bid(0)
    num_programs = ct.num_blocks(0)

    tiles_m = ct.cdiv(M, TILE_M)
    tiles_n = ct.cdiv(N, TILE_N)
    total_tiles = tiles_m * tiles_n
    k_tiles = ct.cdiv(K, TILE_K)
    zero_pad = ct.PaddingMode.ZERO

    # TODO: Implement persistent GEMM
    # Iterate persistently over output tiles assigned to this block.
    # Map each logical tile to matrix coordinates, then run tiled ct.mma GEMM.
    # Keep accumulation in float32 before final cast/store.
    pass


def launch_persistent_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    TILE_M, TILE_N, TILE_K = 128, 128, 64
    C = torch.empty(M, N, dtype=A.dtype, device=A.device)
    occupancy = 2
    tiles_m = math.ceil(M / TILE_M)
    tiles_n = math.ceil(N / TILE_N)
    total_tiles = tiles_m * tiles_n
    num_programs = min(NUM_SM * occupancy, total_tiles)
    grid = (num_programs, 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        ct_persistent_gemm,
        (A, B, C, M, N, K, TILE_M, TILE_N, TILE_K),
    )
    return C


def run_persistent_gemm():
    print("\n=== 10-2: Persistent GEMM ===\n")
    M, N, K = 1024, 1024, 1024
    inputs = {
        "A": torch.randn(M, K, dtype=torch.float16, device="cuda"),
        "B": torch.randn(K, N, dtype=torch.float16, device="cuda"),
    }
    test_puzzle(
        launch_persistent_gemm,
        ref_persistent_gemm,
        inputs,
        label="10-2 Persistent GEMM",
        hint="Use persistent output-tile scheduling and tiled ct.mma accumulation.",
    )


if __name__ == "__main__":
    run_persistent_softmax()
    run_persistent_gemm()
