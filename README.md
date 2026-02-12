# cutile Puzzles

Learn [cutile](https://docs.nvidia.com/cuda/cutile-python) (NVIDIA's Pythonic GPU DSL for Blackwell GPUs) through progressive, hands-on puzzles.

Inspired by [Triton Puzzles](https://github.com/srush/Triton-Puzzles-Lite) and [TileLang Puzzles](https://github.com/tile-ai/tilelang-puzzles).

## What is cutile?

cutile (`cuda.tile`) is a Python DSL that compiles to optimized GPU kernels for NVIDIA Blackwell architecture. You write Python with tile-level operations, and the compiler handles memory hierarchy, pipelining, and hardware mapping automatically.

```python
import cuda.tile as ct

@ct.kernel
def vector_add(a, b, c, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    a_tile = ct.load(a, index=(bid,), shape=(TILE,))
    b_tile = ct.load(b, index=(bid,), shape=(TILE,))
    ct.store(c, index=(bid,), tile=a_tile + b_tile)
```

## Requirements

- NVIDIA Blackwell GPU (B200, RTX 5080, RTX 5090)
- CUDA Toolkit 13.1+
- Python 3.10+
- PyTorch 2.9.1+
- cutile Python (`pip install cuda-tile` or development install)

## Setup

```bash
# Activate the environment with cutile installed
source /root/env/bin/activate

# Verify cutile is available
python -c "import cuda.tile as ct; print('cutile ready!')"
```

## How to Use

Each puzzle is a standalone Python script. Run from the repo root:

```bash
# Run a puzzle (with TODOs for you to fill in)
python puzzles/01-vector-add.py

# Run the reference solution
python ans/01-vector-add.py
```

### Workflow

1. Read the puzzle file — each sub-task has a math specification and hints
2. Fill in the `# TODO` sections in the `@ct.kernel` function
3. Run the puzzle to check correctness
4. Compare with the reference solution in `ans/`

### Test Output

When your implementation is correct:
```
✅ 01-1 Vec Add (load/store): PASS
```

When something is wrong, you get detailed diagnostics:
```
❌ 01-1 Vec Add (load/store): FAIL
   matched: 0/4096 [0.00%]
   max absolute diff: 5.03e+00
   ...
   Hint: Use ct.load(a, index=(bid,), shape=(TILE,)) to load a tile.
```

## Puzzle Progression

### Easy (Puzzles 01-04): Fundamentals

| # | Puzzle | Key Concepts |
|---|--------|-------------|
| 01 | [Vector Add](puzzles/01-vector-add.py) | `ct.load/store`, `ct.gather/scatter`, `ct.bid`, `ct.launch` |
| 02 | [Element-wise Ops](puzzles/02-elementwise-ops.py) | `ct.where`, `ct.exp`, `ct.astype`, `@ct.function` |
| 03 | [Outer Product](puzzles/03-outer-product.py) | 2D grids, `ct.bid(0)/ct.bid(1)`, broadcasting |
| 04 | [Backward Op](puzzles/04-backward-op.py) | Gradients, `ct.atomic_add`, chain rule |

### Medium (Puzzles 05-07): Reductions & Normalization

| # | Puzzle | Key Concepts |
|---|--------|-------------|
| 05 | [Reduce Sum](puzzles/05-reduce-sum.py) | `ct.sum(axis)`, multi-tile accumulation |
| 06 | [Softmax](puzzles/06-softmax.py) | `ct.max`, `ct.exp2`, numerical stability, chunking |
| 07 | [RMSNorm](puzzles/07-rmsnorm.py) | `ct.rsqrt`, normalization, weight scaling |

### Hard (Puzzles 08-11): Matrix Ops & Advanced Patterns

| # | Puzzle | Key Concepts |
|---|--------|-------------|
| 08 | [MatMul](puzzles/08-matmul.py) | `ct.mma` (tensor cores), K-loop, float32 accumulators |
| 09 | [Flash Attention](puzzles/09-flash-attention.py) | Online softmax, streaming QKV, `ct.mma` + attention |
| 10 | [Persistent Scheduling](puzzles/10-persistent-scheduling.py) | `ct.num_blocks`, tile-strided loops, occupancy hints |
| 11 | [Quantized MatMul](puzzles/11-quantized-matmul.py) | FP8 types, per-channel scaling, mixed-precision GEMM |

## cutile Quick Reference

### Two Memory Access Patterns

**Structured (load/store)** — partition array into regular tiles:
```python
tile = ct.load(array, index=(bid,), shape=(TILE,))
ct.store(array, index=(bid,), tile=result)
```

**Indexed (gather/scatter)** — access elements by computed indices:
```python
indices = bid * TILE + ct.arange(TILE, dtype=ct.int32)
tile = ct.gather(array, indices)       # OOB → 0
ct.scatter(array, indices, result)     # OOB → ignored
```

### Key APIs

| Category | Functions |
|----------|----------|
| **Block info** | `ct.bid(dim)`, `ct.num_blocks(dim)`, `ct.num_tiles(array, axis, shape)` |
| **Tile creation** | `ct.full(shape, val, dtype)`, `ct.zeros(shape, dtype)`, `ct.arange(size, dtype)` |
| **Math** | `ct.exp`, `ct.exp2`, `ct.log`, `ct.rsqrt`, `ct.sqrt`, `ct.tanh` |
| **Reduction** | `ct.sum(axis)`, `ct.max(axis)`, `ct.min(axis)` |
| **Comparison** | `ct.where(cond, x, y)`, `ct.greater`, `ct.equal`, ... |
| **Matrix** | `ct.mma(a, b, acc)` (tensor cores), `a @ b` (matmul) |
| **Atomic** | `ct.atomic_add(array, indices, value)` |
| **Types** | `ct.astype(tile, dtype)`, `ct.Constant[int]` |

### Debugging

```python
# Print from inside a kernel (use opt_level=0 for ordered output)
@ct.kernel(opt_level=0)
def debug_kernel(a, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tile = ct.load(a, index=(bid,), shape=(TILE,))
    ct.printf("Block %d: first element = %f\n",
              ct.full((1,), bid, dtype=ct.int32),
              tile[:1].astype(ct.float32))
```

## Repository Structure

```
cutile-puzzles/
├── puzzles/          # Incomplete puzzles (fill in the TODOs)
│   ├── 01-vector-add.py
│   ├── 02-elementwise-ops.py
│   ├── ...
│   └── 11-quantized-matmul.py
├── ans/              # Reference solutions
│   ├── 01-vector-add.py
│   └── ...
├── common/
│   └── utils.py      # test_puzzle() and bench_puzzle() utilities
├── README.md
└── CLAUDE.md         # AI assistant guidance
```

## License

MIT
