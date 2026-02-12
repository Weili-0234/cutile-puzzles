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

Uses online softmax (streaming over KV tiles) for numerical stability:
    1. Each block handles TILE_M query positions.
    2. Initialize per-query accumulators:
        m_i = ct.full((TILE_M,), -inf, ct.float32)   — running max
        l_i = ct.full((TILE_M,), 0.0, ct.float32)    — running sum of exp
        o_i = ct.full((TILE_M,), 0.0, ct.float32)    — running weighted sum
    3. For each KV chunk of size TILE_N:
        a. Load q_tile: (TILE_M,), k_tile: (TILE_N,), v_tile: (TILE_N,).
        b. Compute scores: qk = q_tile[:, None] * k_tile[None, :] — shape (TILE_M, TILE_N).
        c. Scale for exp2: qk = qk * INV_LOG_2
        d. New running max: m_new = max(m_i, ct.max(qk, axis=1))
        e. Correction factor: alpha = ct.exp2(m_i - m_new)
        f. Shifted exp: p = ct.exp2(qk - m_new[:, None])
        g. Update sum: l_i = l_i * alpha + ct.sum(p, axis=1)
        h. Update output: o_i = o_i * alpha + ct.sum(p * v_tile[None, :], axis=1)
        i. m_i = m_new
    4. Final normalization: O = o_i / l_i

Inputs:
    Q: Tensor([M,], float32)
    K: Tensor([N,], float32)
    V: Tensor([N,], float32)

Output:
    O: Tensor([M,], float32)

HINT: Use ct.exp2 with INV_LOG_2 scaling for performance. The key insight
of online softmax is that you can correct previous partial sums when the
running maximum changes, via the alpha = exp2(m_old - m_new) factor.
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
    # 1. Load q_tile at index (bid,) with shape (TILE_M,)
    # 2. Initialize accumulators:
    #    m_i = ct.full((TILE_M,), -math.inf, dtype=ct.float32)  — running max
    #    l_i = ct.full((TILE_M,), 0.0, dtype=ct.float32)        — running exp sum
    #    o_i = ct.full((TILE_M,), 0.0, dtype=ct.float32)        — running output
    # 3. Loop over KV chunks: for j in range(num_tiles_n):
    #    a. Load k_tile at (j,) with shape (TILE_N,)
    #    b. Load v_tile at (j,) with shape (TILE_N,)
    #    c. Compute scores: qk = q_tile[:, None] * k_tile[None, :]  — (TILE_M, TILE_N)
    #    d. Scale: qk = qk * INV_LOG_2
    #    e. Row max: m_new = max(m_i, ct.max(qk, axis=1))
    #    f. Correction: alpha = ct.exp2(m_i - m_new)
    #    g. Exp weights: p = ct.exp2(qk - m_new[:, None])
    #    h. Update: l_i = l_i * alpha + ct.sum(p, axis=1)
    #    i. Update: o_i = o_i * alpha + ct.sum(p * v_tile[None, :], axis=1)
    #    j. m_i = m_new
    # 4. Normalize: result = o_i / l_i
    # 5. Store to o at index (bid,)
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
        hint="Online softmax: track m_i (running max), l_i (running sum of exp), "
        "o_i (running output). Correct old values with alpha = exp2(m_old - m_new).",
    )


# ---------------------------------------------------------------------------
# 09-2: Tiled Flash Attention with ct.mma
# ---------------------------------------------------------------------------
r"""
Full FlashAttention with matrix multiply: Q @ K^T -> softmax -> @ V.

Q: (M, D), K: (N, D), V: (N, D), Output: (M, D)

Algorithm:
    1. Each block handles TILE_M query rows.
    2. Load Q tile: (TILE_M, D) at index (bid, 0). Keep in original dtype for ct.mma.
    3. Compute qk_scale_log2 = qk_scale * INV_LOG_2 (for use with exp2).
    4. Initialize accumulators:
        m_i: (TILE_M, 1) = -inf     — running max per query
        l_i: (TILE_M, 1) = 0.0      — running exp sum per query
        acc: (TILE_M, D) = 0.0       — running output accumulator
    5. For each KV chunk of TILE_N:
        a. Load K tile: (TILE_N, D), transpose to (D, TILE_N).
        b. QK = ct.mma(q_tile, k_transposed, zeros) -> (TILE_M, TILE_N) in float32.
        c. Apply scale after mma (on float32 result):
           m_new = max(m_i, ct.max(QK, axis=-1, keepdims=True) * qk_scale_log2)
           QK = QK * qk_scale_log2 - m_new
        d. Correction: alpha = ct.exp2(m_i - m_new)
        e. Softmax weights: p = ct.exp2(QK)  — already shifted by m_new
        f. Update l_i = l_i * alpha + ct.sum(p, axis=-1, keepdims=True)
        g. Scale acc: acc = acc * alpha
        h. Load V tile: (TILE_N, D).
        i. Cast p to input dtype and accumulate: acc = ct.mma(p_cast, v_tile, acc)
        j. m_i = m_new
    6. Normalize: acc = acc / l_i
    7. Store at (bid, 0).

Inputs:
    Q: Tensor([M, D], float16)
    K: Tensor([N, D], float16)
    V: Tensor([N, D], float16)

Output:
    O: Tensor([M, D], float16)

HINT: To transpose K for QK computation, load K at (j, 0) with shape (TILE_N, D)
and use ct.transpose(k_tile) to get (D, TILE_N). Keep Q in its original dtype for
ct.mma and apply qk_scale to the float32 QK result. The alpha correction factor
broadcasts from (TILE_M, 1) to (TILE_M, D).
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
    # 1. Compute qk_scale_log2 = qk_scale * INV_LOG_2
    # 2. Load q_tile at (bid, 0) with shape (TILE_M, TILE_D) — keep in original dtype
    # 3. Initialize accumulators:
    #    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
    #    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    #    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)
    # 4. Loop over KV chunks: for j in range(num_tiles_n):
    #    a. Load k_tile at (j, 0) with shape (TILE_N, TILE_D)
    #    b. Transpose K: k_t = ct.transpose(k_tile) — gives (TILE_D, TILE_N)
    #    c. Compute QK scores (both inputs are float16, result is float32):
    #       qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
    #       qk = ct.mma(q_tile, k_t, qk) — (TILE_M, TILE_N)
    #    d. Apply scale to float32 QK and compute row max:
    #       m_new = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2)
    #       qk = qk * qk_scale_log2 - m_new  — scale and shift in one step
    #    e. Correction: alpha = ct.exp2(m_i - m_new)
    #    f. Softmax weights: p = ct.exp2(qk) — already shifted by m_new
    #    g. Update exp sum: l_i = l_i * alpha + ct.sum(p, axis=-1, keepdims=True)
    #    h. Scale accumulator: acc = acc * alpha
    #    i. Load v_tile at (j, 0) with shape (TILE_N, TILE_D)
    #    j. Cast p to input dtype: p_cast = p.astype(q_arr.dtype)
    #    k. Accumulate PV: acc = ct.mma(p_cast, v_tile, acc)
    #    l. Update running max: m_i = m_new
    # 5. Normalize: acc = acc / l_i
    # 6. Cast and store: ct.store(o_arr, (bid, 0), acc.astype(o_arr.dtype))
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
        hint="Transpose K with ct.transpose(k_tile). Apply qk_scale * INV_LOG_2 to the "
        "float32 QK result (not to Q). Cast p to input dtype before ct.mma with V.",
    )


if __name__ == "__main__":
    run_scalar_attention()
    run_tiled_attention()
