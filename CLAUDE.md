# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

cutile Puzzles is a progressive tutorial for learning [cutile](https://docs.nvidia.com/cuda/cutile-python) (`cuda.tile`), NVIDIA's Pythonic GPU DSL for Blackwell architecture. It contains 11 puzzles progressing from basic vector operations to Flash Attention and quantized GEMM.

## Repository Structure

- `puzzles/` — Incomplete puzzle templates with `# TODO` markers for the learner to fill in
- `ans/` — Complete reference solutions for all 11 puzzles
- `common/utils.py` — Shared `test_puzzle()` and `bench_puzzle()` utilities

## Running Puzzles

Each puzzle is a standalone script. Run from the repo root:
```bash
source /root/env/bin/activate
python puzzles/01-vector-add.py    # Run puzzle (with TODOs)
python ans/01-vector-add.py        # Run reference solution
```

## cutile Kernel Pattern

Every kernel follows this structure:
```python
import cuda.tile as ct

ConstInt = ct.Constant[int]

@ct.kernel
def my_kernel(input_array, output_array, TILE_SIZE: ConstInt):
    bid = ct.bid(0)                                        # Block index
    tile = ct.load(input_array, index=(bid,), shape=(TILE_SIZE,))  # Load tile
    result = tile * 2                                       # Compute
    ct.store(output_array, index=(bid,), tile=result)       # Store result

# Launch: ct.launch(stream, grid, kernel, args)
ct.launch(torch.cuda.current_stream(), grid, my_kernel, (input, output, TILE_SIZE))
```

## Key cutile Primitives

**Memory access (two paradigms):**
- `ct.load(array, index, shape)` / `ct.store(array, index, tile)` — structured tile access
- `ct.gather(array, indices)` / `ct.scatter(array, indices, tile)` — indexed access with auto OOB handling

**Compute:**
- `ct.mma(a, b, acc)` — matrix multiply-accumulate (tensor cores)
- `ct.sum/max/min(tile, axis, keepdims)` — reductions
- `ct.exp/exp2/log/rsqrt/sqrt/tanh(tile)` — math functions
- `ct.where(cond, true_val, false_val)` — conditional
- `ct.atomic_add(array, indices, value)` — atomic reduction

**Tile creation:**
- `ct.full(shape, value, dtype)` — constant tile
- `ct.arange(size, dtype)` — range [0, 1, ..., size-1]
- `ct.zeros/ones(shape, dtype)` — zero/one tiles

**Block info:**
- `ct.bid(dim)` — block index in dimension
- `ct.num_blocks(dim)` — total blocks in dimension
- `ct.num_tiles(array, axis, shape)` — tiles in partition

**Types:**
- `ct.Constant[int]` — compile-time constant annotation
- `ct.astype(tile, dtype)` / `tile.astype(dtype)` — type conversion
- `ct.float32`, `ct.float16`, `ct.int32`, `ct.bool_` — data types

**Decorators:**
- `@ct.kernel` — GPU entry point (not callable directly; use ct.launch)
- `@ct.kernel(occupancy=N)` — with occupancy hint
- `@ct.function` — reusable device helper function

## Testing Pattern

No test framework — each file has inline `run_*()` functions that call:
- `test_puzzle(ct_fn, ref_fn, inputs, label, hint)` — runs both, compares with torch.allclose
- `bench_puzzle(ct_fn, ref_fn, inputs, bench_torch=True)` — benchmarks against PyTorch

Input tensors are passed as `**kwargs` dicts to both ct_fn and ref_fn.

## Puzzle Progression

01-VectorAdd, 02-ElementwiseOps, 03-OuterProduct, 04-BackwardOp (easy)
→ 05-ReduceSum, 06-Softmax, 07-RMSNorm (medium)
→ 08-MatMul, 09-FlashAttention, 10-PersistentScheduling, 11-QuantizedMatMul (hard)

## When Helping Students Write Kernels

- All tile dimensions must be powers of 2
- Use float32 accumulators for precision, even with float16 inputs
- Use `ct.exp2(x * INV_LOG_2)` instead of `ct.exp(x)` for performance (INV_LOG_2 = 1/log(2))
- `ct.PaddingMode.ZERO` for zero-padded OOB loads; `ct.PaddingMode.NEG_INF` for softmax
- `check_bounds=True` is default on gather/scatter — OOB reads return 0, OOB writes are ignored
- For debugging: use `ct.printf("fmt %f", tile)` with `@ct.kernel(opt_level=0)`
- For persistent scheduling: `for tile_id in range(ct.bid(0), total, ct.num_blocks(0)):`

## Common Mistakes

- Forgetting to cast to float32 before reductions (causes precision loss with float16)
- Not using `padding_mode=ct.PaddingMode.NEG_INF` for softmax (incorrect max from zero-padded OOB)
- Using non-power-of-2 tile dimensions (compile error)
- Forgetting `.astype(output.dtype)` before ct.store (dtype mismatch)
