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

Uses online softmax (streaming over KV tiles) for numerical stability.
"""


def ref_scalar_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    scores = Q[:, None] * K[None, :]  # (M, N)
    attn = torch.softmax(scores, dim=-1)
    return attn @ V


@ct.kernel
def ct_scalar_attention(q, k, v, o, N_val: int, TILE_M: ConstInt, TILE_N: ConstInt):
    bid = ct.bid(0)
    num_tiles_n = ct.cdiv(N_val, TILE_N)

    # Load query tile
    q_tile = ct.load(q, index=(bid,), shape=(TILE_M,), padding_mode=ct.PaddingMode.ZERO)
    q_tile = q_tile.astype(ct.float32)

    # Initialize online softmax accumulators
    m_i = ct.full((TILE_M,), -math.inf, dtype=ct.float32)  # running max
    l_i = ct.full((TILE_M,), 0.0, dtype=ct.float32)        # running exp sum
    o_i = ct.full((TILE_M,), 0.0, dtype=ct.float32)        # running output

    for j in range(num_tiles_n):
        # Load K and V tiles for this chunk
        k_tile = ct.load(k, index=(j,), shape=(TILE_N,), padding_mode=ct.PaddingMode.ZERO)
        k_tile = k_tile.astype(ct.float32)
        v_tile = ct.load(v, index=(j,), shape=(TILE_N,), padding_mode=ct.PaddingMode.ZERO)
        v_tile = v_tile.astype(ct.float32)

        # Compute scores: (TILE_M, TILE_N) via broadcast
        qk = q_tile[:, None] * k_tile[None, :]

        # Scale for exp2
        qk = qk * INV_LOG_2

        # Online softmax update
        m_new = max(m_i, ct.max(qk, axis=1))          # (TILE_M,)
        alpha = ct.exp2(m_i - m_new)                    # correction factor
        p = ct.exp2(qk - m_new[:, None])                # (TILE_M, TILE_N)

        # Update running sum and output
        l_i = l_i * alpha + ct.sum(p, axis=1)
        o_i = o_i * alpha + ct.sum(p * v_tile[None, :], axis=1)
        m_i = m_new

    # Final normalization
    result = o_i / l_i
    ct.store(o, index=(bid,), tile=result)


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
    )
    bench_puzzle(launch_scalar_attention, ref_scalar_attention, inputs, bench_torch=True)


# ---------------------------------------------------------------------------
# 09-2: Tiled Flash Attention with ct.mma
# ---------------------------------------------------------------------------
r"""
Full FlashAttention with matrix multiply: Q @ K^T -> softmax -> @ V.

Q: (M, D), K: (N, D), V: (N, D), Output: (M, D)

Uses online softmax with ct.mma for tensor core acceleration.
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

    # Combine qk_scale and log2 conversion for use with exp2
    qk_scale_log2 = qk_scale * INV_LOG_2

    # Load Q tile: (TILE_M, TILE_D) — keep in original dtype for ct.mma
    q_tile = ct.load(q_arr, index=(bid, 0), shape=(TILE_M, TILE_D), padding_mode=ct.PaddingMode.ZERO)

    # Initialize online softmax accumulators
    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)  # running max per query row
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)        # running exp sum per query row
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)    # running output accumulator

    for j in range(num_tiles_n):
        # Load K tile and transpose for QK^T: (TILE_N, TILE_D) -> (TILE_D, TILE_N)
        k_tile = ct.load(k_arr, index=(j, 0), shape=(TILE_N, TILE_D), padding_mode=ct.PaddingMode.ZERO)
        k_t = ct.transpose(k_tile)  # (TILE_D, TILE_N)

        # Compute QK^T using tensor cores: (TILE_M, TILE_D) @ (TILE_D, TILE_N) -> (TILE_M, TILE_N)
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q_tile, k_t, qk)

        # Online softmax update — apply scale to the float32 QK result
        # Moving qk_scale multiplication after reduce_max improves performance
        m_new = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2)  # (TILE_M, 1)
        qk = qk * qk_scale_log2 - m_new  # scale and shift in one step
        alpha = ct.exp2(m_i - m_new)                            # correction factor
        p = ct.exp2(qk)                                         # (TILE_M, TILE_N)

        # Update running exp sum
        l_i = l_i * alpha + ct.sum(p, axis=-1, keepdims=True)

        # Scale existing accumulator by correction factor
        acc = acc * alpha

        # Load V tile: (TILE_N, TILE_D)
        v_tile = ct.load(v_arr, index=(j, 0), shape=(TILE_N, TILE_D), padding_mode=ct.PaddingMode.ZERO)

        # Accumulate P @ V using tensor cores
        p_cast = p.astype(q_arr.dtype)
        acc = ct.mma(p_cast, v_tile, acc)  # (TILE_M, TILE_D)

        m_i = m_new

    # Final normalization
    acc = acc / l_i

    # Cast and store
    acc = acc.astype(o_arr.dtype)
    ct.store(o_arr, index=(bid, 0), tile=acc)


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
    )
    bench_puzzle(launch_tiled_attention, ref_tiled_attention, inputs, bench_torch=True)


if __name__ == "__main__":
    run_scalar_attention()
    run_tiled_attention()
