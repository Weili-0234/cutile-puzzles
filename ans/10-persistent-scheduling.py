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

Pattern:
    for row_idx in range(ct.bid(0), n_rows, ct.num_blocks(0)):
        ... process row_idx ...

Each block starts at its own bid and strides by the total number of blocks,
ensuring all rows are covered without conflicts.

Uses gather/scatter for single-tile-per-row softmax with -inf padding for OOB.

Inputs:
    A: Tensor([M, N], float16)

Output:
    B: Tensor([M, N], float16)  where B[i,:] = softmax(A[i,:])
"""


def ref_persistent_softmax(A: torch.Tensor) -> torch.Tensor:
    return torch.softmax(A.float(), dim=1).to(A.dtype)


@ct.kernel(occupancy=4)
def ct_persistent_softmax(a, b, n_rows: ConstInt, TILE_N: ConstInt):
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)
    offsets = ct.arange(TILE_N, dtype=ct.int32)

    for row_idx in range(pid, n_rows, num_programs):
        # Load row using gather with -inf padding for OOB
        row = ct.gather(a, (row_idx, offsets), check_bounds=True, padding_value=-math.inf)

        # Cast to float32 for numerical stability
        row = ct.astype(row, ct.float32)

        # Subtract max for numerical stability
        row_max = ct.max(row, 0, keepdims=True)
        shifted = ct.sub(row, row_max)

        # Exponentiate
        numerator = ct.exp(shifted)

        # Sum for normalization
        denominator = ct.sum(numerator, 0, keepdims=True)

        # Divide to get softmax
        result = ct.truediv(numerator, denominator)

        # Cast back and store
        result = ct.astype(result, a.dtype)
        ct.scatter(b, (row_idx, offsets), result, check_bounds=True)


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
    )
    bench_puzzle(launch_persistent_softmax, ref_persistent_softmax, inputs, bench_torch=True)


# ---------------------------------------------------------------------------
# 10-2: Persistent GEMM
# ---------------------------------------------------------------------------
r"""
Matrix multiplication C = A @ B with persistent scheduling.

Instead of a 2D grid (one block per output tile), use a 1D grid where
each block processes multiple output tiles in a persistent loop.

Pattern:
    bid = ct.bid(0)
    num_programs = ct.num_blocks(0)
    tiles_m = cdiv(M, TILE_M)
    tiles_n = cdiv(N, TILE_N)
    total_tiles = tiles_m * tiles_n
    for tile_id in range(bid, total_tiles, num_programs):
        tile_m = tile_id // tiles_n
        tile_n = tile_id % tiles_n
        ... standard GEMM inner loop ...

Inputs:
    A: Tensor([M, K], float16)
    B: Tensor([K, N], float16)

Output:
    C: Tensor([M, N], float16)  where C = A @ B
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

    for tile_id in range(bid, total_tiles, num_programs):
        # Map 1D tile_id to 2D tile coordinates
        bid_m = tile_id // tiles_n
        bid_n = tile_id % tiles_n

        # Initialize accumulator
        accumulator = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)

        # K-dimension loop
        for k in range(k_tiles):
            a_tile = ct.load(
                a, index=(bid_m, k), shape=(TILE_M, TILE_K), padding_mode=zero_pad
            )
            b_tile = ct.load(
                b, index=(k, bid_n), shape=(TILE_K, TILE_N), padding_mode=zero_pad
            )
            accumulator = ct.mma(a_tile, b_tile, accumulator)

        # Cast to output dtype and store
        result = ct.astype(accumulator, c.dtype)
        ct.store(c, index=(bid_m, bid_n), tile=result)


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
    )
    bench_puzzle(launch_persistent_gemm, ref_persistent_gemm, inputs, bench_torch=True)


if __name__ == "__main__":
    run_persistent_softmax()
    run_persistent_gemm()
