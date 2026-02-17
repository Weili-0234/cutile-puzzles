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

# Allow running from any directory
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
Think in tiles: each block owns a row tile of `C`, streams across K-chunks,
accumulates partial dot-products in float32, then writes the output tile.

Inputs:
    A: Tensor([M, K], float16)
    B: Tensor([K,], float16)

Output:
    C: Tensor([M,], float16)  where C[i] = sum_k(A[i, k] * B[k])

HINT: Structure GEMV as tiled dot products over K with float32 accumulation.
"""


def ref_gemv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (A.float() @ B.float()).to(A.dtype)


@ct.kernel
def ct_gemv(a, b, c, TILE_M: ConstInt, TILE_K: ConstInt):
    bid = ct.bid(0)
    num_tiles_k = ct.num_tiles(a, axis=1, shape=(TILE_M, TILE_K))

    # TODO: Implement GEMV (matrix-vector multiply)
    # Iterate over K tiles, accumulate each row's dot-product in float32,
    # and write the final TILE_M outputs for this block.
    # Handle boundary tiles safely when loading.
    pass


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
    test_puzzle(
        launch_gemv,
        ref_gemv,
        inputs,
        label="08-1 GEMV",
        hint="Treat GEMV as tiled row-wise dot products over K in float32.",
    )


# ---------------------------------------------------------------------------
# 08-2: Naive GEMM (Matrix-Matrix Multiply, no ct.mma)
# ---------------------------------------------------------------------------
r"""
Compute C = A @ B using tiled matrix multiplication without tensor cores.

Each block computes a (TILE_M, TILE_N) output tile. A 2D grid maps blocks
to output positions. The K dimension is iterated in chunks of TILE_K.

Instead of tensor core instructions, we use the Python @ operator on tiles
(ct.matmul), which still performs a tiled matrix multiply but without
hardware MMA acceleration.
Each block computes one output tile. Accumulate across K-tiles in float32 and
store after the full reduction.

Inputs:
    A: Tensor([M, K], float16)
    B: Tensor([K, N], float16)

Output:
    C: Tensor([M, N], float16)  where C[i, j] = sum_k(A[i, k] * B[k, j])

HINT: Keep a float32 accumulator per output tile and reduce across K tiles.
"""


def ref_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (A.float() @ B.float()).to(A.dtype)


@ct.kernel
def ct_gemm_naive(a, b, c, TILE_M: ConstInt, TILE_N: ConstInt, TILE_K: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    num_tiles_k = ct.num_tiles(a, axis=1, shape=(TILE_M, TILE_K))

    # TODO: Implement naive GEMM (no ct.mma)
    # Compute one output tile per block by reducing over K tiles.
    # Use float32 accumulation before casting to output dtype.
    # Boundary tiles should remain safe under padding.
    pass


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
    test_puzzle(
        launch_gemm_naive,
        ref_gemm,
        inputs,
        label="08-2 Naive GEMM",
        hint="Use tiled K-reduction with a float32 accumulator for each output tile.",
    )


# ---------------------------------------------------------------------------
# 08-3: GEMM with ct.mma (Tensor Cores)
# ---------------------------------------------------------------------------
r"""
Compute C = A @ B using ct.mma for tensor core acceleration.

ct.mma(a, b, acc) computes acc += a @ b using hardware matrix-multiply-accumulate
instructions (tensor cores). The accumulator must be float32; input tiles can be
float16 or bfloat16.
As in naive GEMM, each block computes one output tile and reduces over K.
The key difference is using `ct.mma` for tensor-core accumulation.

Inputs:
    A: Tensor([M, K], float16)
    B: Tensor([K, N], float16)

Output:
    C: Tensor([M, N], float16)  where C[i, j] = sum_k(A[i, k] * B[k, j])

HINT: Use `ct.mma` in the K-reduction loop with a float32 accumulator tile.
"""


@ct.kernel
def ct_gemm_mma(a, b, c, TILE_M: ConstInt, TILE_N: ConstInt, TILE_K: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    num_tiles_k = ct.num_tiles(a, axis=1, shape=(TILE_M, TILE_K))

    # TODO: Implement GEMM with ct.mma (tensor cores)
    # Follow tiled GEMM structure, but use ct.mma for K-chunk accumulation.
    # Keep the accumulator in float32 and cast only for final store.
    # Handle edge tiles via safe load behavior.
    pass


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
    test_puzzle(
        launch_gemm_mma,
        ref_gemm,
        inputs,
        label="08-3 GEMM (ct.mma)",
        hint="Use ct.mma for K-tile accumulation and keep accumulator precision high.",
    )


if __name__ == "__main__":
    run_gemv()
    run_gemm_naive()
    run_gemm_mma()
