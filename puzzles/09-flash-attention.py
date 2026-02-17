"""
Puzzle 09: Flash Attention
============================
Implement Flash Attention using cutile, from a simplified scalar version
to the full tiled version with ct.mma and online softmax.

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

# Constant for converting natural log to log base 2
INV_LOG_2 = 1.0 / math.log(2)


# ---------------------------------------------------------------------------
# 09-1: Scalar Flash Attention (no ct.mma)
# ---------------------------------------------------------------------------
r"""
Simplified 1D attention: for each query position m, compute weighted
attention over all key/value positions.

    score[n] = Q[m] * K[n]
    attn = softmax(score)
    O[m] = sum_n(attn[n] * V[n])

Q: (M,), K: (N,), V: (N,), Output: (M,)

Use online softmax while streaming KV chunks. Track running statistics in
float32 so previously accumulated values remain valid when the local maximum
changes from one chunk to the next.

Inputs:
    Q: Tensor([M,], float32)
    K: Tensor([N,], float32)
    V: Tensor([N,], float32)

Output:
    O: Tensor([M,], float32)

HINT: Focus on maintaining stable running softmax statistics across KV chunks.
"""


def ref_scalar_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    scores = Q[:, None] * K[None, :]  # (M, N)
    attn = torch.softmax(scores, dim=-1)
    return attn @ V


@ct.kernel
def ct_scalar_attention(q, k, v, o, N_val: int, TILE_M: ConstInt, TILE_N: ConstInt):
    bid = ct.bid(0)
    num_tiles_n = ct.cdiv(N_val, TILE_N)

    # TODO: Implement scalar flash attention with online softmax
    # Compute scalar flash attention with online softmax across KV chunks.
    # Maintain running max/sum/output state in float32 for stability.
    # Normalize once all chunks for this query tile are processed.
    pass


def launch_scalar_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    M = Q.shape[0]
    N = K.shape[0]
    TILE_M = 32
    TILE_N = 64
    O = torch.empty(M, dtype=Q.dtype, device=Q.device)
    grid = (math.ceil(M / TILE_M), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        ct_scalar_attention,
        (Q, K, V, O, N, TILE_M, TILE_N),
    )
    return O


def run_scalar_attention():
    print("\n=== 09-1: Scalar Flash Attention (no ct.mma) ===\n")
    M, N = 256, 512
    inputs = {
        "Q": torch.randn(M, dtype=torch.float32, device="cuda"),
        "K": torch.randn(N, dtype=torch.float32, device="cuda"),
        "V": torch.randn(N, dtype=torch.float32, device="cuda"),
    }
    test_puzzle(
        launch_scalar_attention,
        ref_scalar_attention,
        inputs,
        label="09-1 Scalar Flash Attention",
        hint="Use online softmax state updates so chunked attention stays stable.",
    )


# ---------------------------------------------------------------------------
# 09-2: Tiled Flash Attention with ct.mma
# ---------------------------------------------------------------------------
r"""
Full FlashAttention with matrix multiply: Q @ K^T -> softmax -> @ V.

Q: (M, D), K: (N, D), V: (N, D), Output: (M, D)
Each block handles a query tile, computes tiled QK scores, applies online
softmax over KV chunks, accumulates PV in float32, then normalizes to produce
the output tile.

Inputs:
    Q: Tensor([M, D], float16)
    K: Tensor([N, D], float16)
    V: Tensor([N, D], float16)

Output:
    O: Tensor([M, D], float16)

HINT: Keep QK/PV accumulation high precision and preserve online-softmax state
consistency across chunks.
"""


def ref_tiled_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(
        Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)
    ).squeeze(0)


@ct.kernel
def ct_tiled_attention(
    q_arr, k_arr, v_arr, o_arr,
    N_val: int,
    qk_scale: float,
    TILE_M: ConstInt, TILE_N: ConstInt, TILE_D: ConstInt,
):
    bid = ct.bid(0)
    num_tiles_n = ct.cdiv(N_val, TILE_N)

    # TODO: Implement tiled flash attention with ct.mma
    # Implement tiled flash attention with online softmax and ct.mma.
    # Keep running statistics and output accumulation in float32.
    # Normalize at the end and store in output dtype.
    pass


def launch_tiled_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    M, D = Q.shape
    N = K.shape[0]
    TILE_M = 64
    TILE_N = 64
    TILE_D = D
    qk_scale = 1.0 / math.sqrt(D)
    O = torch.empty(M, D, dtype=Q.dtype, device=Q.device)
    grid = (math.ceil(M / TILE_M), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        ct_tiled_attention,
        (Q, K, V, O, N, qk_scale, TILE_M, TILE_N, TILE_D),
    )
    return O


def run_tiled_attention():
    print("\n=== 09-2: Tiled Flash Attention (ct.mma) ===\n")
    M, N, D = 256, 256, 64
    inputs = {
        "Q": torch.randn(M, D, dtype=torch.float16, device="cuda"),
        "K": torch.randn(N, D, dtype=torch.float16, device="cuda"),
        "V": torch.randn(N, D, dtype=torch.float16, device="cuda"),
    }
    test_puzzle(
        launch_tiled_attention,
        ref_tiled_attention,
        inputs,
        label="09-2 Tiled Flash Attention (ct.mma)",
        hint="Combine tiled QK/PV matmuls with online-softmax state updates in float32.",
    )


if __name__ == "__main__":
    run_scalar_attention()
    run_tiled_attention()
