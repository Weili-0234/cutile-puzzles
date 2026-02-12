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

Algorithm:
    1. bid = ct.bid(0) selects which TILE_M rows this block handles.
    2. Initialize accumulator: ct.full((TILE_M,), 0.0, dtype=ct.float32).
    3. Loop over K in chunks (num_tiles_k iterations):
        a. Load A tile: shape (TILE_M, TILE_K) at index (bid, k).
        b. Load B tile: shape (TILE_K,) at index (k,).
        c. Cast both to float32.
        d. Broadcast B across rows and multiply: prod = a_tile * b_tile[None, :]
           — this gives (TILE_M, TILE_K).
        e. Sum along axis=1 to get partial result of shape (TILE_M,).
        f. Add to accumulator.
    4. Cast accumulator to output dtype and store.

Inputs:
    A: Tensor([M, K], float16)
    B: Tensor([K,], float16)

Output:
    C: Tensor([M,], float16)  where C[i] = sum_k(A[i, k] * B[k])

HINT: Use ct.load with padding_mode=ct.PaddingMode.ZERO. Broadcast B
with b_tile[None, :] to shape (1, TILE_K) before multiplying with A tile.
"""


def ref_gemv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (A.float() @ B.float()).to(A.dtype)


@ct.kernel
def ct_gemv(a, b, c, TILE_M: ConstInt, TILE_K: ConstInt):
    bid = ct.bid(0)
    num_tiles_k = ct.num_tiles(a, axis=1, shape=(TILE_M, TILE_K))

    # TODO: Implement GEMV (matrix-vector multiply)
    # 1. Initialize accumulator: ct.full((TILE_M,), 0.0, dtype=ct.float32)
    # 2. Loop over K dimension: for k in range(num_tiles_k):
    #    a. Load A tile at (bid, k) with shape (TILE_M, TILE_K), padding_mode=ct.PaddingMode.ZERO
    #    b. Load B tile at (k,) with shape (TILE_K,), padding_mode=ct.PaddingMode.ZERO
    #    c. Cast both tiles to float32 using .astype(ct.float32)
    #    d. Multiply: a_tile * b_tile[None, :] — broadcasts (TILE_K,) to (1, TILE_K)
    #    e. Sum along axis=1: ct.sum(product, axis=1) — gives shape (TILE_M,)
    #    f. Add partial sum to accumulator
    # 3. Cast accumulator to output dtype: acc.astype(c.dtype)
    # 4. Store to c at index (bid,)
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
        hint="Broadcast b_tile[None, :] to (1, TILE_K), multiply with a_tile, "
        "then ct.sum along axis=1 to reduce K.",
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

Algorithm:
    1. bid_m = ct.bid(0), bid_n = ct.bid(1) — 2D block indices.
    2. Compute num_tiles_k = ct.num_tiles(A, axis=1, shape=(TILE_M, TILE_K)).
    3. Initialize accumulator: ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32).
    4. Loop over K chunks:
        a. Load A tile: shape (TILE_M, TILE_K) at index (bid_m, k).
        b. Load B tile: shape (TILE_K, TILE_N) at index (k, bid_n).
        c. Cast both to float32.
        d. Accumulate: acc = acc + (a_tile @ b_tile).
    5. Cast to output dtype and store at (bid_m, bid_n).

Inputs:
    A: Tensor([M, K], float16)
    B: Tensor([K, N], float16)

Output:
    C: Tensor([M, N], float16)  where C[i, j] = sum_k(A[i, k] * B[k, j])

HINT: The @ operator on cutile tiles performs matrix multiplication.
a_tile @ b_tile with shapes (TILE_M, TILE_K) @ (TILE_K, TILE_N) gives (TILE_M, TILE_N).
"""


def ref_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (A.float() @ B.float()).to(A.dtype)


@ct.kernel
def ct_gemm_naive(a, b, c, TILE_M: ConstInt, TILE_N: ConstInt, TILE_K: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    num_tiles_k = ct.num_tiles(a, axis=1, shape=(TILE_M, TILE_K))

    # TODO: Implement naive GEMM (no ct.mma)
    # 1. Initialize accumulator: ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
    # 2. Loop over K: for k in range(num_tiles_k):
    #    a. Load A tile at (bid_m, k) with shape (TILE_M, TILE_K),
    #       padding_mode=ct.PaddingMode.ZERO
    #    b. Load B tile at (k, bid_n) with shape (TILE_K, TILE_N),
    #       padding_mode=ct.PaddingMode.ZERO
    #    c. Cast both tiles to float32
    #    d. Matrix multiply and accumulate: acc = acc + (a_tile @ b_tile)
    # 3. Cast accumulator to output dtype: acc.astype(c.dtype)
    # 4. Store to c at index (bid_m, bid_n)
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
        hint="Load A at (bid_m, k), B at (k, bid_n). Use @ operator: acc = acc + (a_tile @ b_tile).",
    )


# ---------------------------------------------------------------------------
# 08-3: GEMM with ct.mma (Tensor Cores)
# ---------------------------------------------------------------------------
r"""
Compute C = A @ B using ct.mma for tensor core acceleration.

ct.mma(a, b, acc) computes acc += a @ b using hardware matrix-multiply-accumulate
instructions (tensor cores). The accumulator must be float32; input tiles can be
float16 or bfloat16.

Algorithm:
    1. bid_m = ct.bid(0), bid_n = ct.bid(1) — 2D block indices.
    2. Compute num_tiles_k = ct.num_tiles(A, axis=1, shape=(TILE_M, TILE_K)).
    3. Initialize accumulator: ct.full((TILE_M, TILE_N), 0, dtype=ct.float32).
    4. Loop over K chunks:
        a. Load A tile: shape (TILE_M, TILE_K) at index (bid_m, k).
        b. Load B tile: shape (TILE_K, TILE_N) at index (k, bid_n).
        c. MMA: accumulator = ct.mma(a_tile, b_tile, accumulator).
    5. Cast to output dtype and store at (bid_m, bid_n).

Inputs:
    A: Tensor([M, K], float16)
    B: Tensor([K, N], float16)

Output:
    C: Tensor([M, N], float16)  where C[i, j] = sum_k(A[i, k] * B[k, j])

HINT: ct.mma(a_tile, b_tile, accumulator) — a_tile is (M, K), b_tile is (K, N),
accumulator is (M, N) in float32. No need to cast input tiles — ct.mma handles it.
"""


@ct.kernel
def ct_gemm_mma(a, b, c, TILE_M: ConstInt, TILE_N: ConstInt, TILE_K: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    num_tiles_k = ct.num_tiles(a, axis=1, shape=(TILE_M, TILE_K))

    # TODO: Implement GEMM with ct.mma (tensor cores)
    # 1. Initialize accumulator: ct.full((TILE_M, TILE_N), 0, dtype=ct.float32)
    # 2. Loop over K: for k in range(num_tiles_k):
    #    a. Load A tile at (bid_m, k) with shape (TILE_M, TILE_K),
    #       padding_mode=ct.PaddingMode.ZERO
    #    b. Load B tile at (k, bid_n) with shape (TILE_K, TILE_N),
    #       padding_mode=ct.PaddingMode.ZERO
    #    c. Tensor core MMA: accumulator = ct.mma(a_tile, b_tile, accumulator)
    # 3. Cast to output dtype: ct.astype(accumulator, c.dtype)
    # 4. Store to c at index (bid_m, bid_n)
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
        hint="ct.mma(a_tile, b_tile, accumulator) — no need to cast inputs. "
        "Accumulator must be float32.",
    )


if __name__ == "__main__":
    run_gemv()
    run_gemm_naive()
    run_gemm_mma()
