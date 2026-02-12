# Findings: Cutile Puzzles Design Research

## Codebase Analysis

### cutile (cuda.tile) Programming Model
- **Tile-centric**: All computation operates on tiles loaded from global memory
- **Key APIs**: `ct.load/store` (tile-based), `ct.gather/scatter` (index-based), `ct.mma` (matrix multiply-accumulate)
- **Decorators**: `@ct.kernel` for GPU entry points, `@ct.function` for device functions
- **Constants**: `ct.Constant[int]` type annotations for compile-time values
- **Block indexing**: `ct.bid(dim)`, `ct.num_blocks(dim)`
- **Tile space**: `ct.num_tiles(array, axis, shape)` for partition queries
- **Launch**: `ct.launch(stream, grid, kernel, args)`
- **No explicit shared memory**: Compiler manages memory hierarchy automatically
- **TMA**: Hardware-accelerated memory transfers, enabled by default in load/store

### cutile vs tilelang - Key Differences
| Feature | cutile | tilelang |
|---------|--------|----------|
| Memory access | `ct.load/store`, `ct.gather/scatter` | `T.copy`, direct indexing |
| Matrix ops | `ct.mma(a, b, acc)` | `T.gemm(A, B, C)` |
| Block indexing | `ct.bid(dim)` | Context manager `with T.Kernel(...) as (bx, by)` |
| Shared memory | Implicit (compiler-managed) | Explicit `T.alloc_shared` |
| Registers | Implicit (tiles are in registers) | `T.alloc_fragment` |
| Pipelining | Manual loops | `T.Pipelined(range, stages)` |
| Constants | `ct.Constant[int]` annotation | `T.const("name")` |
| Tile creation | `ct.full/zeros/ones/arange` | `T.fill/T.clear` |
| Reductions | `ct.sum/max/min(axis)` | `T.reduce_sum/max/min` |
| Scheduling | Manual persistent loops | `T.Persistent` construct |

### tilelang-puzzles Structure (Reference)
- **10 puzzles**: copy -> vec_add -> outer_add -> backward -> reduce -> softmax -> flash_attn -> matrix -> conv -> dequant_mm
- **Format**: Single .py file per puzzle with docstring spec, ref PyTorch fn, incomplete TileLang kernel, test harness
- **TODO markers**: Students fill in `# TODO` sections in kernel body
- **Solutions**: Complete implementations in `ans/` folder
- **Testing**: `test_puzzle()` compares against PyTorch reference with `torch.allclose`
- **Difficulty**: Easy (1-4), Medium (5-7), Hard (8-10)
- **No external test framework**: Each file is standalone, run with `python3 puzzles/XX-name.py`

### TileGym Kernels (cutile reference implementations)
- **Element-wise**: dropout, silu_and_mul, swiglu
- **Normalization**: rms_norm, softmax (3 variants)
- **Positional**: rope
- **Attention**: fmha (prefill/decode), mla (prefill/decode/split-kv)
- **Matrix**: matmul, bmm, group_gemm
- **Patterns**: Static persistent scheduling, online softmax, recomputation backward, autotuning

## Key cutile Patterns Worth Teaching

### Pattern 1: Tile Load/Store (Fundamental)
```python
tile = ct.load(array, index=(bid,), shape=(TILE_SIZE,))
ct.store(array, index=(bid,), tile=result)
```

### Pattern 2: Gather/Scatter (Flexible Indexing)
```python
indices = bid * TILE + ct.arange(TILE, dtype=ct.int32)
tile = ct.gather(array, indices, padding_value=0)
ct.scatter(array, indices, result)
```

### Pattern 3: 2D Tiling with Load/Store
```python
bid_x, bid_y = ct.bid(0), ct.bid(1)
tile = ct.load(A, (bid_x, bid_y), shape=(TILE_M, TILE_N))
```

### Pattern 4: Reduction
```python
row_max = ct.max(tile, axis=-1, keepdims=True)
row_sum = ct.sum(tile, axis=-1, keepdims=True)
```

### Pattern 5: Matrix Multiply-Accumulate
```python
acc = ct.full((M, N), 0, dtype=ct.float32)
for k in range(num_k_tiles):
    a = ct.load(A, (bid_m, k), shape=(M, K))
    b = ct.load(B, (k, bid_n), shape=(K, N))
    acc = ct.mma(a, b, acc)
```

### Pattern 6: Online Softmax (for Attention)
```python
m_i = ct.full((M, 1), float('-inf'), ct.float32)
l_i = ct.full((M, 1), 0, ct.float32)
# streaming update of max, sum, and weighted accumulator
```

### Pattern 7: Static Persistent Scheduling
```python
for tile_id in range(ct.bid(0), total_tiles, ct.num_blocks(0)):
    # process tile_id
```

---

## Debugging Capabilities Comparison

### cutile debugging tools
| Tool | Usage | Notes |
|------|-------|-------|
| `ct.printf("fmt %d", tile)` | C-style device printf | Output interleaved across blocks; use `opt_level=0` to fix ordering |
| `ct.assert_(cond, "msg")` | Device-side assertion | Terminates kernel on failure; significant overhead |
| `check_bounds=True` | Default on gather/scatter | OOB reads→padding_value, OOB writes→ignored |
| `opt_level=0` | `@ct.kernel(opt_level=0)` | Disable optimizations for debugging |
| `CUDA_TILE_LOGS=CUTILEIR` | Env var | Print IR to stderr |
| `CUDA_TILE_DUMP_TILEIR=dir` | Env var | Dump MLIR to files |
| `CUDA_TILE_ENABLE_CRASH_DUMP=1` | Env var | Dump crash artifacts to .zip |
| **NO interpreter mode** | - | Unlike Triton, cutile ALWAYS runs on GPU |

### Triton Puzzles debugging UX (what makes it great)
1. **TRITON_INTERPRET=1** — CPU execution, no GPU needed, full introspection
2. **print() in kernels** — Works natively in interpreter mode as teaching tool
3. **Memory access validation** — Catches OOB per-block, shows byte offsets + invalid masks
4. **Layered error reporting**: emoji → side-by-side values → launch params → memory diagnostics
5. **Demos before puzzles** — Use print() to show indexing patterns visually

### TileGym testing UX (what's great)
1. **Detailed mismatch report**: matched %, value ranges, max diff, mismatched indices
2. **Auto-detected tolerances** by dtype (fp16→1e-2, fp32→1e-5, etc.)
3. **Per-tensor tolerance** via TestParam wrapper
4. **CI regression detection** with ±5% threshold

### Gap analysis for cutile-puzzles
| Triton feature | cutile equivalent | Impact |
|---------------|-------------------|--------|
| Interpreter mode (CPU) | None | Cannot run without GPU; no CPU fallback |
| `print()` in kernel | `ct.printf()` with `opt_level=0` | Works but noisier; C-style format strings |
| Memory OOB detection | `check_bounds=True` (default) | Automatic; OOB silently handled, not reported |
| Per-block access logs | None | Cannot trace which block had issues |
| `print_log=True` verbose | Custom in test harness | We must build this ourselves |
