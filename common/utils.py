"""
Utilities for cutile puzzles.

Provides test_puzzle() and bench_puzzle() for validating and benchmarking
cutile kernel implementations against PyTorch reference functions.
"""

import math
from typing import Callable, Optional

import cuda.tile as ct
import torch


def get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """Auto-detect rtol/atol based on dtype."""
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-2, 2e-2
    elif dtype == torch.float32:
        return 1e-4, 1e-5
    elif dtype == torch.float64:
        return 1e-12, 1e-15
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return 1e-1, 1e-1
    else:
        # Integer types: exact match
        return 0, 0


def test_puzzle(
    ct_fn: Callable,
    ref_fn: Callable,
    inputs: dict[str, torch.Tensor],
    label: str = "Puzzle",
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    hint: Optional[str] = None,
    print_log: bool = False,
):
    """
    Test a cutile puzzle solution against a PyTorch reference.

    Args:
        ct_fn: Callable that takes input tensors and returns the cutile result.
              This is typically a launch wrapper (not the raw kernel).
        ref_fn: PyTorch reference function.
        inputs: Dict of input tensor name → tensor. These are passed as kwargs
                to both ct_fn and ref_fn.
        label: Display name for the test.
        atol: Absolute tolerance. If None, auto-detected from output dtype.
        rtol: Relative tolerance. If None, auto-detected from output dtype.
        hint: Optional hint message to show on failure.
        print_log: If True, always print detailed diagnostics (even on pass).
    """
    # Make copies so the kernel doesn't modify reference inputs
    inputs_ref = {k: v.clone() for k, v in inputs.items()}
    inputs_ct = {k: v.clone() for k, v in inputs.items()}

    # Run reference
    output_ref = ref_fn(**inputs_ref)

    # Run cutile
    output_ct = ct_fn(**inputs_ct)

    # Auto-detect tolerances if not specified
    if atol is None or rtol is None:
        auto_rtol, auto_atol = get_tolerances(output_ref.dtype)
        if atol is None:
            atol = auto_atol
        if rtol is None:
            rtol = auto_rtol

    # Compare
    match = torch.allclose(output_ct, output_ref, atol=atol, rtol=rtol)
    match_emoji = "\u2705" if match else "\u274c"
    print(f"{match_emoji} {label}: {'PASS' if match else 'FAIL'}")

    if not match or print_log:
        # Layer 1: Statistics
        total = output_ref.numel()
        close_mask = torch.isclose(output_ref, output_ct, atol=atol, rtol=rtol)
        matched = close_mask.sum().item()
        pct = 100.0 * matched / total if total > 0 else 0.0
        abs_diff = torch.abs(output_ref.float() - output_ct.float())
        max_abs = abs_diff.max().item()
        mean_abs = abs_diff.mean().item()

        print(f"   matched: {matched}/{total} [{pct:.2f}%]")
        print(f"   max absolute diff: {max_abs:.6e}")
        print(f"   mean absolute diff: {mean_abs:.6e}")
        print(f"   tolerances: atol={atol}, rtol={rtol}")

        # Layer 2: Side-by-side (truncated for large tensors)
        if total <= 256:
            print(f"\n   Yours:  dtype={output_ct.dtype}  shape={tuple(output_ct.shape)}")
            print(f"   {output_ct}")
            print(f"   Spec:   dtype={output_ref.dtype}  shape={tuple(output_ref.shape)}")
            print(f"   {output_ref}")
        else:
            # Show shape/dtype and a small slice
            print(f"\n   Yours:  dtype={output_ct.dtype}  shape={tuple(output_ct.shape)}")
            print(f"   Spec:   dtype={output_ref.dtype}  shape={tuple(output_ref.shape)}")
            print(f"   ref range:  {output_ref.min().item():.6e} : {output_ref.max().item():.6e}")
            print(f"   test range: {output_ct.min().item():.6e} : {output_ct.max().item():.6e}")

        # Layer 3: Mismatched indices (up to 10)
        if not match:
            mismatch_indices = torch.where(~close_mask)
            if len(mismatch_indices[0]) > 0:
                n_show = min(10, len(mismatch_indices[0]))
                indices_list = []
                for i in range(n_show):
                    idx = tuple(dim[i].item() for dim in mismatch_indices)
                    indices_list.append(idx)
                print(f"\n   First {n_show} mismatched indices: {indices_list}")
                # Show values at those indices
                for idx in indices_list[:3]:
                    ref_val = output_ref[idx].item()
                    ct_val = output_ct[idx].item()
                    print(f"     [{idx}]: yours={ct_val:.6e}  spec={ref_val:.6e}  diff={abs(ct_val - ref_val):.6e}")

        # Layer 4: Hint
        if not match and hint:
            print(f"\n   Hint: {hint}")

    return match


def bench_puzzle(
    ct_fn: Callable,
    ref_fn: Optional[Callable],
    inputs: dict[str, torch.Tensor],
    bench_name: str = "cutile",
    bench_torch: bool = False,
    warmups: int = 10,
    repeats: int = 100,
):
    """
    Benchmark a cutile puzzle solution, optionally comparing against PyTorch.

    Args:
        ct_fn: Callable cutile implementation (launch wrapper).
        ref_fn: Optional PyTorch reference for comparison timing.
        inputs: Dict of input tensors.
        bench_name: Label for the cutile benchmark.
        bench_torch: If True, also benchmark the PyTorch reference.
        warmups: Number of warmup iterations.
        repeats: Number of timed iterations.
    """
    if bench_torch and ref_fn is not None:
        # Benchmark PyTorch
        for _ in range(warmups):
            ref_fn(**inputs)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(repeats):
            ref_fn(**inputs)
        end.record()
        torch.cuda.synchronize()
        torch_time = start.elapsed_time(end) / repeats
        print(f"  PyTorch time: {torch_time:.3f} ms")

    # Benchmark cutile
    for _ in range(warmups):
        ct_fn(**inputs)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeats):
        ct_fn(**inputs)
    end.record()
    torch.cuda.synchronize()
    ct_time = start.elapsed_time(end) / repeats
    print(f"  {bench_name} time: {ct_time:.3f} ms")
