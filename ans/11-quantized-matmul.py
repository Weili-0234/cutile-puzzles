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

Implementation strategy:
    1. Use a standard 2D tiled GEMM with ct.mma (fp8 inputs, fp32 accumulator).
    2. After the K-loop, load the scale factors for the current tile's rows/columns.
    3. Apply scaling: result = accumulator * scale_a[:, None] * scale_b[None, :]
       where scale_a is gathered as (TILE_M,) and broadcast, and similarly for scale_b.
    4. Cast to output dtype (bfloat16) and store.

Inputs:
    A_fp8:   Tensor([M, K], float8_e4m3fn)
    B_fp8:   Tensor([K, N], float8_e4m3fn)
    scale_A: Tensor([M],    float32)  — per-row scale for A
    scale_B: Tensor([N],    float32)  — per-column scale for B

Output:
    C: Tensor([M, N], bfloat16)
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

    # Initialize float32 accumulator
    accumulator = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)

    # K-loop: ct.mma supports fp8 inputs with fp32 accumulator
    for k in range(k_tiles):
        a_tile = ct.load(a, index=(bid_m, k), shape=(TILE_M, TILE_K), padding_mode=zero_pad)
        b_tile = ct.load(b, index=(k, bid_n), shape=(TILE_K, TILE_N), padding_mode=zero_pad)
        accumulator = ct.mma(a_tile, b_tile, accumulator)

    # Load per-channel scale factors
    row_offsets = bid_m * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    col_offsets = bid_n * TILE_N + ct.arange(TILE_N, dtype=ct.int32)
    sa = ct.gather(scale_a, row_offsets).reshape((TILE_M, 1))   # (TILE_M, 1)
    sb = ct.gather(scale_b, col_offsets).reshape((1, TILE_N))   # (1, TILE_N)

    # Apply scaling with proper broadcasting over (TILE_M, TILE_N)
    accumulator = accumulator * sa * sb

    # Cast to output dtype and store
    result = ct.astype(accumulator, c.dtype)
    ct.store(c, index=(bid_m, bid_n), tile=result)


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
    )
    bench_puzzle(launch_fp8_matmul, ref_fp8_matmul, inputs, bench_torch=True)


if __name__ == "__main__":
    run_fp8_matmul()
