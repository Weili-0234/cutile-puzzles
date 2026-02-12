"""
Puzzle 08: Matrix Multiplication
==================================
Learn tiled matrix multiplication using cutile.
Start with GEMV (matrix-vector), then naive GEMM, then GEMM with ct.mma (tensor cores).

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


# ---------------------------------------------------------------------------
# 08-1: GEMV (Matrix-Vector Multiply)
# ---------------------------------------------------------------------------
r"""
Compute a matrix-vector product: C[i] = sum_k(A[i, k] * B[k])

Each block processes TILE_M rows of the output. The K dimension is
iterated in chunks of TILE_K, with partial products accumulated
in float32 for numerical stability.

Inputs:
    A: Tensor([M, K], float16)
    B: Tensor([K,], float16)

Output:
    C: Tensor([M,], float16)  where C[i] = sum_k(A[i, k] * B[k])
"""


def ref_gemv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (A.float() @ B.float()).to(A.dtype)


@ct.kernel
def ct_gemv(a, b, c, TILE_M: ConstInt, TILE_K: ConstInt):
    bid = ct.bid(0)
    num_tiles_k = ct.num_tiles(a, axis=1, shape=(TILE_M, TILE_K))

    # Initialize float32 accumulator for TILE_M output rows
    acc = ct.full((TILE_M,), 0.0, dtype=ct.float32)

    for k in range(num_tiles_k):
        # Load A tile: (TILE_M, TILE_K)
        a_tile = ct.load(a, index=(bid, k), shape=(TILE_M, TILE_K), padding_mode=ct.PaddingMode.ZERO)
        a_tile = a_tile.astype(ct.float32)

        # Load B tile: (TILE_K,)
        b_tile = ct.load(b, index=(k,), shape=(TILE_K,), padding_mode=ct.PaddingMode.ZERO)
        b_tile = b_tile.astype(ct.float32)

        # Broadcast B across rows and multiply
        product = a_tile * b_tile[None, :]  # (TILE_M, TILE_K)

        # Sum along K axis to get partial result per row
        partial = ct.sum(product, axis=1)  # (TILE_M,)

        # Accumulate
        acc = acc + partial

    # Cast and store
    acc = acc.astype(c.dtype)
    ct.store(c, index=(bid,), tile=acc)


def launch_gemv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    TILE_M = 32
    TILE_K = 64
    C = torch.empty(M, dtype=A.dtype, device=A.device)
    grid = (math.ceil(M / TILE_M), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, ct_gemv, (A, B, C, TILE_M, TILE_K))
    return C


def run_gemv():
    print("\n=== 08-1: GEMV (Matrix-Vector Multiply) ===\n")
    M, K = 1024, 1024
    inputs = {
        "A": torch.randn(M, K, dtype=torch.float16, device="cuda"),
        "B": torch.randn(K, dtype=torch.float16, device="cuda"),
    }
    test_puzzle(launch_gemv, ref_gemv, inputs, label="08-1 GEMV")
    bench_puzzle(launch_gemv, ref_gemv, inputs, bench_torch=True)


# ---------------------------------------------------------------------------
# 08-2: Naive GEMM (Matrix-Matrix Multiply, no ct.mma)
# ---------------------------------------------------------------------------
r"""
Compute C = A @ B using tiled matrix multiplication without tensor cores.

Each block computes a (TILE_M, TILE_N) output tile. A 2D grid maps blocks
to output positions. The K dimension is iterated in chunks of TILE_K.

Inputs:
    A: Tensor([M, K], float16)
    B: Tensor([K, N], float16)

Output:
    C: Tensor([M, N], float16)  where C[i, j] = sum_k(A[i, k] * B[k, j])
"""


def ref_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (A.float() @ B.float()).to(A.dtype)


@ct.kernel
def ct_gemm_naive(a, b, c, TILE_M: ConstInt, TILE_N: ConstInt, TILE_K: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    num_tiles_k = ct.num_tiles(a, axis=1, shape=(TILE_M, TILE_K))

    # Initialize float32 accumulator
    acc = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)

    for k in range(num_tiles_k):
        # Load A tile: (TILE_M, TILE_K)
        a_tile = ct.load(a, index=(bid_m, k), shape=(TILE_M, TILE_K), padding_mode=ct.PaddingMode.ZERO)
        a_tile = a_tile.astype(ct.float32)

        # Load B tile: (TILE_K, TILE_N)
        b_tile = ct.load(b, index=(k, bid_n), shape=(TILE_K, TILE_N), padding_mode=ct.PaddingMode.ZERO)
        b_tile = b_tile.astype(ct.float32)

        # Matrix multiply and accumulate using @ operator
        acc = acc + (a_tile @ b_tile)

    # Cast to output dtype and store
    acc = acc.astype(c.dtype)
    ct.store(c, index=(bid_m, bid_n), tile=acc)


def launch_gemm_naive(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    TILE_M, TILE_N, TILE_K = 32, 32, 32
    C = torch.empty(M, N, dtype=A.dtype, device=A.device)
    grid = (math.ceil(M / TILE_M), math.ceil(N / TILE_N), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        ct_gemm_naive,
        (A, B, C, TILE_M, TILE_N, TILE_K),
    )
    return C


def run_gemm_naive():
    print("\n=== 08-2: Naive GEMM (no ct.mma) ===\n")
    M, N, K = 512, 512, 512
    inputs = {
        "A": torch.randn(M, K, dtype=torch.float16, device="cuda"),
        "B": torch.randn(K, N, dtype=torch.float16, device="cuda"),
    }
    test_puzzle(launch_gemm_naive, ref_gemm, inputs, label="08-2 Naive GEMM")
    bench_puzzle(launch_gemm_naive, ref_gemm, inputs, bench_torch=True)


# ---------------------------------------------------------------------------
# 08-3: GEMM with ct.mma (Tensor Cores)
# ---------------------------------------------------------------------------
r"""
Compute C = A @ B using ct.mma for tensor core acceleration.

ct.mma(a, b, acc) computes acc += a @ b using hardware matrix-multiply-accumulate
instructions (tensor cores). The accumulator must be float32; input tiles can be
float16 or bfloat16.

Inputs:
    A: Tensor([M, K], float16)
    B: Tensor([K, N], float16)

Output:
    C: Tensor([M, N], float16)  where C[i, j] = sum_k(A[i, k] * B[k, j])
"""


@ct.kernel
def ct_gemm_mma(a, b, c, TILE_M: ConstInt, TILE_N: ConstInt, TILE_K: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    num_tiles_k = ct.num_tiles(a, axis=1, shape=(TILE_M, TILE_K))

    # Initialize float32 accumulator
    accumulator = ct.full((TILE_M, TILE_N), 0, dtype=ct.float32)

    for k in range(num_tiles_k):
        # Load A tile: (TILE_M, TILE_K)
        a_tile = ct.load(a, index=(bid_m, k), shape=(TILE_M, TILE_K), padding_mode=ct.PaddingMode.ZERO)

        # Load B tile: (TILE_K, TILE_N)
        b_tile = ct.load(b, index=(k, bid_n), shape=(TILE_K, TILE_N), padding_mode=ct.PaddingMode.ZERO)

        # Tensor core MMA: accumulator += a_tile @ b_tile
        accumulator = ct.mma(a_tile, b_tile, accumulator)

    # Cast to output dtype and store
    accumulator = ct.astype(accumulator, c.dtype)
    ct.store(c, index=(bid_m, bid_n), tile=accumulator)


def launch_gemm_mma(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    TILE_M, TILE_N, TILE_K = 128, 128, 64
    C = torch.empty(M, N, dtype=A.dtype, device=A.device)
    grid = (math.ceil(M / TILE_M), math.ceil(N / TILE_N), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        ct_gemm_mma,
        (A, B, C, TILE_M, TILE_N, TILE_K),
    )
    return C


def run_gemm_mma():
    print("\n=== 08-3: GEMM with ct.mma (Tensor Cores) ===\n")
    M, N, K = 1024, 1024, 1024
    inputs = {
        "A": torch.randn(M, K, dtype=torch.float16, device="cuda"),
        "B": torch.randn(K, N, dtype=torch.float16, device="cuda"),
    }
    test_puzzle(launch_gemm_mma, ref_gemm, inputs, label="08-3 GEMM (ct.mma)")
    bench_puzzle(launch_gemm_mma, ref_gemm, inputs, bench_torch=True)


if __name__ == "__main__":
    run_gemv()
    run_gemm_naive()
    run_gemm_mma()
