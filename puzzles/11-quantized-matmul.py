"""
Puzzle 11: Quantized Matrix Multiplication
============================================
Learn FP8 per-channel quantized matrix multiplication. The inputs are stored
in float8_e4m3fn with per-channel scale factors. The kernel accumulates in
float32 and applies scaling after the K-loop.

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
# 11-1: FP8 Per-Channel Quantized GEMM
# ---------------------------------------------------------------------------
r"""
Compute C = dequant(A_fp8) @ dequant(B_fp8) with per-channel scaling.

Mathematically:
    C[i, j] = scale_A[i] * scale_B[j] * sum_k(A_fp8[i, k] * B_fp8[k, j])

The key insight is that scale factors can be applied AFTER the matmul:
compute unscaled FP8 GEMM in float32 first, then apply per-row/per-column
scales to the output tile before storing.

Inputs:
    A_fp8:   Tensor([M, K], float8_e4m3fn)
    B_fp8:   Tensor([K, N], float8_e4m3fn)
    scale_A: Tensor([M],    float32)  -- per-row scale for A
    scale_B: Tensor([N],    float32)  -- per-column scale for B

Output:
    C: Tensor([M, N], bfloat16)

HINT: Separate concerns: accumulate GEMM in fp32, then apply per-channel scales.
"""


def ref_fp8_matmul(
    A_fp8: torch.Tensor,
    B_fp8: torch.Tensor,
    scale_A: torch.Tensor,
    scale_B: torch.Tensor,
) -> torch.Tensor:
    # Dequantize and compute in float32
    A_f32 = A_fp8.float() * scale_A[:, None]
    B_f32 = B_fp8.float() * scale_B[None, :]
    return (A_f32 @ B_f32).to(torch.bfloat16)


@ct.kernel
def ct_fp8_matmul(
    a, b, c,
    scale_a, scale_b,
    TILE_M: ConstInt, TILE_N: ConstInt, TILE_K: ConstInt,
):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    K = a.shape[1]
    k_tiles = ct.cdiv(K, TILE_K)
    zero_pad = ct.PaddingMode.ZERO

    # TODO: Implement FP8 per-channel quantized GEMM
    # Run tiled fp8 GEMM with float32 accumulation.
    # After K-reduction, apply per-row and per-column scales to the tile.
    # Cast to output dtype only at final store.
    pass


def launch_fp8_matmul(
    A_fp8: torch.Tensor,
    B_fp8: torch.Tensor,
    scale_A: torch.Tensor,
    scale_B: torch.Tensor,
) -> torch.Tensor:
    M, K = A_fp8.shape
    K2, N = B_fp8.shape
    assert K == K2
    TILE_M, TILE_N, TILE_K = 128, 128, 64
    C = torch.empty(M, N, dtype=torch.bfloat16, device=A_fp8.device)
    grid = (math.ceil(M / TILE_M), math.ceil(N / TILE_N), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        ct_fp8_matmul,
        (A_fp8, B_fp8, C, scale_A, scale_B, TILE_M, TILE_N, TILE_K),
    )
    return C


def run_fp8_matmul():
    print("\n=== 11-1: FP8 Per-Channel Quantized GEMM ===\n")
    M, N, K = 1024, 1024, 1024

    # Generate FP8 test data with proper quantization
    A_fp32 = torch.randn(M, K, device="cuda")
    B_fp32 = torch.randn(K, N, device="cuda")

    # Compute per-channel scales (FP8 e4m3 max is ~448)
    scale_A = A_fp32.abs().amax(dim=1) / 448.0
    scale_B = B_fp32.abs().amax(dim=0) / 448.0

    # Avoid zero scales
    scale_A = scale_A.clamp(min=1e-12)
    scale_B = scale_B.clamp(min=1e-12)

    # Quantize to FP8
    A_fp8 = (A_fp32 / scale_A[:, None]).to(torch.float8_e4m3fn)
    B_fp8 = (B_fp32 / scale_B[None, :]).to(torch.float8_e4m3fn)

    inputs = {
        "A_fp8": A_fp8,
        "B_fp8": B_fp8,
        "scale_A": scale_A,
        "scale_B": scale_B,
    }
    test_puzzle(
        launch_fp8_matmul,
        ref_fp8_matmul,
        inputs,
        label="11-1 FP8 Per-Channel Quantized GEMM",
        hint="Accumulate fp8 matmul in float32 first, then apply per-channel scaling.",
    )


if __name__ == "__main__":
    run_fp8_matmul()
