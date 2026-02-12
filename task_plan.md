# Task Plan: Cutile Puzzles Repository

## Goal
Create a beginner-friendly cutile puzzle repository (11 puzzles) that teaches NVIDIA's cutile (cuda.tile) Python DSL progressively, inspired by tilelang-puzzles and Triton-Puzzles-Lite.

## Status: COMPLETE ✅

---

## Finalized Design Decisions

### Puzzles (11 total, 3 difficulty tiers)

| # | Name | Difficulty | Sub-tasks | Key Concepts |
|---|------|-----------|-----------|--------------|
| 01 | Vector Add | Easy | 3: 1D load/store, 1D gather/scatter, 2D load/store | `ct.load`, `ct.store`, `ct.gather`, `ct.scatter`, `ct.bid`, `ct.launch`, `ct.arange` |
| 02 | Element-wise Ops | Easy | 3: ReLU, fused mul+ReLU, SiLU | `ct.where`, `ct.mul`, `ct.exp`, `ct.astype`, `@ct.function` hint |
| 03 | Outer Product | Easy | 2: 2D load/store, 2D gather/scatter broadcasting | 2D `ct.bid(0)/ct.bid(1)`, `[:, None]` broadcasting, 2D grid |
| 04 | Backward Op | Easy | 2: broadcast mul+ReLU fwd, gradient backward | `ct.greater`, `ct.atomic_add`, gradient computation |
| 05 | Reduce Sum | Medium | 2: single-tile row sum, multi-tile chunked sum | `ct.sum(axis)`, accumulation loop, `ct.full` |
| 06 | Softmax | Medium | 2: single-tile softmax, chunked softmax | `ct.max`, `ct.exp2`, `INV_LOG_2`, numerical stability |
| 07 | RMSNorm | Medium | 2: basic RMSNorm, RMSNorm with weight | `ct.rsqrt`, normalization, weight scaling |
| 08 | MatMul | Hard | 3: GEMV, naive GEMM, GEMM with ct.mma | `ct.mma`, K-loop, float32 accumulators, `ct.num_tiles` |
| 09 | Flash Attention | Hard | 2: scalar flash attn, tiled with ct.mma | Online softmax, streaming QKV, causal masking |
| 10 | Persistent Scheduling | Hard | 2: persistent softmax, persistent GEMM | `ct.num_blocks`, tile-strided loops, occupancy |
| 11 | Quantized MatMul | Hard | 1: FP8 per-channel GEMM | FP8 types, per-channel scaling, `ct.astype` |

### Format
- Progressive scaffolding: easy puzzles give structure, hard puzzles give less
- Each file: docstring spec → PyTorch ref → kernel with TODO → launch wrapper (given) → test harness
- Solutions in `ans/` directory
- Multiple sub-tasks per puzzle (Option A)
- Both load/store and gather/scatter shown in puzzles 1-2, hints on APIs in puzzles 3+

### Testing Infrastructure
- `common/utils.py` with `test_puzzle()` and `bench_puzzle()`
- Layered error output: emoji → stats → side-by-side → mismatched indices → puzzle hints
- Auto-detected tolerances by dtype
- Launch wrappers given to students (not TODOs)

### Debugging UX
- `ct.printf` demos in early puzzles with `opt_level=0`
- `check_bounds=True` default on gather/scatter (auto boundary handling)
- Puzzle-specific failure hints
- Rich error output compensating for no interpreter mode

---

## Implementation Phases

### Phase 1: Infrastructure ✅
- [x] Create directory structure (puzzles/, ans/, common/)
- [x] Implement common/utils.py (test_puzzle, bench_puzzle)
- [x] Create README.md
- [x] Create CLAUDE.md

### Phase 2: Easy Puzzles (01-04) ✅
- [x] 01-vector-add.py (puzzle + ans) — 3 sub-tasks, all pass
- [x] 02-elementwise-ops.py (puzzle + ans) — 3 sub-tasks, all pass
- [x] 03-outer-product.py (puzzle + ans) — 2 sub-tasks, all pass
- [x] 04-backward-op.py (puzzle + ans) — 2 sub-tasks, all pass

### Phase 3: Medium Puzzles (05-07) ✅
- [x] 05-reduce-sum.py (puzzle + ans) — 2 sub-tasks, all pass
- [x] 06-softmax.py (puzzle + ans) — 2 sub-tasks, all pass
- [x] 07-rmsnorm.py (puzzle + ans) — 2 sub-tasks, all pass

### Phase 4: Hard Puzzles (08-11) ✅
- [x] 08-matmul.py (puzzle + ans) — 3 sub-tasks, all pass
- [x] 09-flash-attention.py (puzzle + ans) — 2 sub-tasks, all pass
- [x] 10-persistent-scheduling.py (puzzle + ans) — 2 sub-tasks, all pass
- [x] 11-quantized-matmul.py (puzzle + ans) — 1 sub-task, pass

### Phase 5: Polish ✅
- [x] End-to-end testing of all puzzles (24/24 sub-tasks pass on Blackwell GPU)
- [x] Documentation review (README.md, CLAUDE.md)
- [x] Final README with setup instructions
