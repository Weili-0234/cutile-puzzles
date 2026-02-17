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
Use one row tile per block, compute normalization in float32, and scale the
input row by the inverse root-mean-square. Use the real row length `N` for
the mean, not the tile size.

Inputs:
    A: Tensor([M, N], float16)

Output:
    B: Tensor([M, N], float16)  where B[i,:] = A[i,:] / rms(A[i,:])

HINT: Keep the normalization path in float32 and apply epsilon before rsqrt.
"""


def ref_rmsnorm(A: torch.Tensor) -> torch.Tensor:
    x = A.float()
    rms = x.pow(2).mean(-1, keepdim=True).add_(1e-6).rsqrt()
    return (x * rms).to(A.dtype)


@ct.kernel
def ct_rmsnorm(a, b, N_VAL: ConstInt, TILE_N: ConstInt, EPS: ConstFloat):
    bid = ct.bid(0)

    # TODO: Implement basic RMSNorm
    # Compute row RMS in float32, then normalize the row tile.
    # Use N_VAL for the mean and include EPS before rsqrt.
    # Cast back to output dtype when storing.
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
        hint="Compute RMS in float32, apply eps-stabilized rsqrt, then normalize.",
    )


# ---------------------------------------------------------------------------
# 07-2: RMSNorm with Learned Weight
# ---------------------------------------------------------------------------
r"""
Compute RMSNorm with a learned weight vector:
B[i, :] = (A[i, :] / rms(A[i, :])) * W[:]

Same as 07-1 but the normalized output is multiplied element-wise by a
weight vector W of shape (N,).
This keeps the same RMS computation, then applies a learned weight vector
element-wise to the normalized row.

Inputs:
    A: Tensor([M, N], float16)
    W: Tensor([N,], float16)

Output:
    B: Tensor([M, N], float16)  where B[i,:] = (A[i,:] / rms(A[i,:])) * W[:]

HINT: Reuse the RMS path from 07-1, then apply the learned weight per feature.
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
    # Implement RMSNorm as in 07-1, then apply per-feature weight.
    # Keep normalization and weighting math in float32.
    # Cast to output dtype before storing.
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
        hint="Use 07-1 RMSNorm flow, then apply learned per-feature scaling.",
    )


if __name__ == "__main__":
    run_rmsnorm()
    run_rmsnorm_weighted()
