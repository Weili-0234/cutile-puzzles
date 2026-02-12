"""
Puzzle 07: RMSNorm
===================
Learn how to implement Root Mean Square Layer Normalization in cutile.
RMSNorm is widely used in modern transformer architectures (e.g., LLaMA).

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
ConstFloat = ct.Constant[float]


# ---------------------------------------------------------------------------
# 07-1: Basic RMSNorm
# ---------------------------------------------------------------------------
r"""
Compute RMSNorm for each row: B[i, :] = A[i, :] / rms(A[i, :])

where rms(x) = sqrt(mean(x^2) + eps)

Algorithm:
    1. Use ct.bid(0) to get the row index (one block per row).
    2. Load the row as a (1, TILE_N) tile from A.
       Use padding_mode=ct.PaddingMode.ZERO so OOB elements are zero.
    3. Cast to float32 for precision: ct.astype(tile, ct.float32).
    4. Square: x_sq = tile * tile.
    5. Mean of squares: mean_sq = ct.sum(x_sq, axis=1, keepdims=True) / N.
       (Divide by N, not TILE_N, to get the correct mean over actual elements.)
    6. Add epsilon: mean_sq_eps = mean_sq + eps.
    7. Inverse sqrt: rstd = ct.rsqrt(mean_sq_eps).
    8. Normalize: result = tile * rstd.
    9. Cast back to original dtype and store.

Inputs:
    A: Tensor([M, N], float16)

Output:
    B: Tensor([M, N], float16)  where B[i,:] = A[i,:] / rms(A[i,:])

HINT: ct.rsqrt computes 1/sqrt(x). Use ct.sum with axis=1 and divide by N
for the mean. Don't forget to add eps before rsqrt.
"""


def ref_rmsnorm(A: torch.Tensor) -> torch.Tensor:
    x = A.float()
    rms = x.pow(2).mean(-1, keepdim=True).add_(1e-6).rsqrt()
    return (x * rms).to(A.dtype)


@ct.kernel
def ct_rmsnorm(a, b, N_VAL: ConstInt, TILE_N: ConstInt, EPS: ConstFloat):
    bid = ct.bid(0)

    # TODO: Implement basic RMSNorm
    # 1. Load (1, TILE_N) tile from a at (bid, 0)
    #    with padding_mode=ct.PaddingMode.ZERO
    # 2. Cast to float32: ct.astype(tile, ct.float32)
    # 3. Square: x_sq = tile * tile
    # 4. Sum of squares: sum_sq = ct.sum(x_sq, axis=1, keepdims=True)
    # 5. Mean: mean_sq = sum_sq / N_VAL
    # 6. Add eps: mean_sq_eps = mean_sq + EPS
    # 7. Inverse sqrt: rstd = ct.rsqrt(mean_sq_eps)
    # 8. Normalize: result = tile * rstd
    # 9. Cast back: ct.astype(result, ct.float16)
    # 10. Store to b at (bid, 0)
    pass


def launch_rmsnorm(A: torch.Tensor) -> torch.Tensor:
    M, N = A.shape
    TILE_N = 1024
    EPS = 1e-6
    B = torch.empty_like(A)
    grid = (M, 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        ct_rmsnorm,
        (A, B, N, TILE_N, EPS),
    )
    return B


def run_rmsnorm():
    print("\n=== 07-1: Basic RMSNorm ===\n")
    M, N = 256, 1024
    inputs = {
        "A": torch.randn(M, N, dtype=torch.float16, device="cuda"),
    }
    test_puzzle(
        launch_rmsnorm,
        ref_rmsnorm,
        inputs,
        label="07-1 RMSNorm (basic)",
        hint="Compute mean of squares with ct.sum / N, then ct.rsqrt(mean_sq + eps). "
        "Multiply input by rstd to normalize.",
    )


# ---------------------------------------------------------------------------
# 07-2: RMSNorm with Learned Weight
# ---------------------------------------------------------------------------
r"""
Compute RMSNorm with a learned weight vector:
B[i, :] = (A[i, :] / rms(A[i, :])) * W[:]

Same as 07-1 but the normalized output is multiplied element-wise by a
weight vector W of shape (N,).

Algorithm:
    1-8. Same as 07-1 (load, cast, square, mean, rsqrt, normalize).
    9.  Load weight: w = ct.load(W, (0,), shape=(TILE_N,)).
        Reshape to (1, TILE_N) for broadcasting.
    10. Multiply: result = normalized * w.
    11. Cast back and store.

Inputs:
    A: Tensor([M, N], float16)
    W: Tensor([N,], float16)

Output:
    B: Tensor([M, N], float16)  where B[i,:] = (A[i,:] / rms(A[i,:])) * W[:]

HINT: Load the weight vector with ct.load(W, (0,), shape=(TILE_N,)) and
reshape to (1, TILE_N) for broadcasting with the (1, TILE_N) normalized tile.
"""


def ref_rmsnorm_weighted(A: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    x = A.float()
    rms = x.pow(2).mean(-1, keepdim=True).add_(1e-6).rsqrt()
    return (x * rms * W.float()).to(A.dtype)


@ct.kernel
def ct_rmsnorm_weighted(
    a, b, w, N_VAL: ConstInt, TILE_N: ConstInt, EPS: ConstFloat
):
    bid = ct.bid(0)

    # TODO: Implement RMSNorm with learned weight
    # 1. Load (1, TILE_N) tile from a at (bid, 0) with ZERO padding
    # 2. Cast to float32
    # 3. Square and compute mean: sum(x^2) / N_VAL
    # 4. Add eps and rsqrt
    # 5. Normalize: tile * rstd
    # 6. Load weight: ct.load(w, (0,), shape=(TILE_N,))
    # 7. Reshape weight to (1, TILE_N) for broadcasting
    # 8. Cast weight to float32
    # 9. Multiply normalized by weight
    # 10. Cast back to float16 and store to b at (bid, 0)
    pass


def launch_rmsnorm_weighted(A: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    M, N = A.shape
    TILE_N = 1024
    EPS = 1e-6
    B = torch.empty_like(A)
    grid = (M, 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        ct_rmsnorm_weighted,
        (A, B, W, N, TILE_N, EPS),
    )
    return B


def run_rmsnorm_weighted():
    print("\n=== 07-2: RMSNorm with Weight ===\n")
    M, N = 256, 1024
    inputs = {
        "A": torch.randn(M, N, dtype=torch.float16, device="cuda"),
        "W": torch.randn(N, dtype=torch.float16, device="cuda"),
    }
    test_puzzle(
        launch_rmsnorm_weighted,
        ref_rmsnorm_weighted,
        inputs,
        label="07-2 RMSNorm (weighted)",
        hint="Same as 07-1 but also load W with ct.load(w, (0,), shape=(TILE_N,)). "
        "Reshape to (1, TILE_N) and multiply with normalized result.",
    )


if __name__ == "__main__":
    run_rmsnorm()
    run_rmsnorm_weighted()
