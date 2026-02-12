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

Inputs:
    A: Tensor([M, N], float16)

Output:
    B: Tensor([M, N], float16)  where B[i,:] = A[i,:] / rms(A[i,:])
"""


def ref_rmsnorm(A: torch.Tensor) -> torch.Tensor:
    x = A.float()
    rms = x.pow(2).mean(-1, keepdim=True).add_(1e-6).rsqrt()
    return (x * rms).to(A.dtype)


@ct.kernel
def ct_rmsnorm(a, b, N_VAL: ConstInt, TILE_N: ConstInt, EPS: ConstFloat):
    bid = ct.bid(0)

    # Load one row as a 2D tile
    tile = ct.load(
        a, index=(bid, 0), shape=(1, TILE_N),
        padding_mode=ct.PaddingMode.ZERO,
    )

    # Cast to float32 for precision
    tile = ct.astype(tile, ct.float32)

    # Square each element
    x_sq = tile * tile

    # Sum of squares along row
    sum_sq = ct.sum(x_sq, axis=1, keepdims=True)

    # Mean of squares (divide by actual N, not TILE_N)
    mean_sq = sum_sq / N_VAL

    # Add epsilon for numerical stability
    mean_sq_eps = mean_sq + EPS

    # Inverse square root
    rstd = ct.rsqrt(mean_sq_eps)

    # Normalize
    result = tile * rstd

    # Cast back and store
    result = ct.astype(result, ct.float16)
    ct.store(b, index=(bid, 0), tile=result)


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
    test_puzzle(launch_rmsnorm, ref_rmsnorm, inputs, label="07-1 RMSNorm (basic)")
    bench_puzzle(launch_rmsnorm, ref_rmsnorm, inputs, bench_torch=True)


# ---------------------------------------------------------------------------
# 07-2: RMSNorm with Learned Weight
# ---------------------------------------------------------------------------
r"""
Compute RMSNorm with a learned weight vector:
B[i, :] = (A[i, :] / rms(A[i, :])) * W[:]

Same as 07-1 but the normalized output is multiplied element-wise by a
weight vector W of shape (N,).

Inputs:
    A: Tensor([M, N], float16)
    W: Tensor([N,], float16)

Output:
    B: Tensor([M, N], float16)  where B[i,:] = (A[i,:] / rms(A[i,:])) * W[:]
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

    # Load one row
    tile = ct.load(
        a, index=(bid, 0), shape=(1, TILE_N),
        padding_mode=ct.PaddingMode.ZERO,
    )

    # Cast to float32
    tile = ct.astype(tile, ct.float32)

    # Compute RMS normalization
    x_sq = tile * tile
    sum_sq = ct.sum(x_sq, axis=1, keepdims=True)
    mean_sq = sum_sq / N_VAL
    mean_sq_eps = mean_sq + EPS
    rstd = ct.rsqrt(mean_sq_eps)
    normalized = tile * rstd

    # Load weight vector and reshape for broadcasting
    weight = ct.load(w, index=(0,), shape=(TILE_N,))
    weight = weight.reshape((1, TILE_N))
    weight = ct.astype(weight, ct.float32)

    # Apply weight
    result = normalized * weight

    # Cast back and store
    result = ct.astype(result, ct.float16)
    ct.store(b, index=(bid, 0), tile=result)


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
        launch_rmsnorm_weighted, ref_rmsnorm_weighted, inputs,
        label="07-2 RMSNorm (weighted)",
    )
    bench_puzzle(launch_rmsnorm_weighted, ref_rmsnorm_weighted, inputs, bench_torch=True)


if __name__ == "__main__":
    run_rmsnorm()
    run_rmsnorm_weighted()
