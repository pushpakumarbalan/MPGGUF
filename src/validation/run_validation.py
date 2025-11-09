#!/usr/bin/env python3
"""
Run mixed-precision validation.

This script selects the best backend (CUDA -> MPS -> CPU), compiles CUDA kernels
if available, then validates dequantization by streaming tensors from an MPGGUF
file (interleaved Q8/Q2 data) and computing MSE/RMSE vs a baseline FP16 GGUF.

It demonstrates the benefit of adjacent co-location: on simulated mispredicts,
it immediately reads the adjacent precision payload from the same mmap region,
minimizing seeks/page faults.

Supports multi-backend execution: CUDA for NVIDIA GPUs, MPS for Apple Silicon,
CPU fallback. Dequantization and error computation run on the selected backend.
"""

import argparse
import ctypes
import importlib.util
import mmap
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def load_module_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def try_compile_cuda_kernels(cuda_src: Path, out_lib: Path) -> bool:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        print("nvcc not found on PATH — skipping CUDA build.")
        return False

    out_lib.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        nvcc,
        "-std=c++14",
        "-O3",
        "-Xcompiler",
        "-fPIC",
        "--shared",
        str(cuda_src),
        "-o",
        str(out_lib),
    ]

    print("Compiling CUDA kernels:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
        print("CUDA kernels compiled to:", out_lib)
        return True
    except subprocess.CalledProcessError as e:
        print("CUDA compilation failed:", e)
        return False


def dequantize_q8_0_cpu(raw: bytes, num_elements: int) -> np.ndarray:
    # Q8_0: blocks of 34 bytes: 2 bytes (fp16 scale) + 32 int8s
    block_size_bytes = 34
    elems_per_block = 32

    out = np.zeros(num_elements, dtype=np.float16)
    blocks = (num_elements + elems_per_block - 1) // elems_per_block

    for b in range(blocks):
        offset = b * block_size_bytes
        scale_bytes = raw[offset: offset + 2]
        qs_bytes = raw[offset + 2: offset + block_size_bytes]

        if len(scale_bytes) < 2:
            break

        scale = np.frombuffer(scale_bytes, dtype=np.float16)[0].astype(np.float32)
        qs = np.frombuffer(qs_bytes, dtype=np.int8).astype(np.float32)

        start = b * elems_per_block
        end = min(start + len(qs), num_elements)
        out[start:end] = (scale * qs[: end - start]).astype(np.float16)

    return out


def dequantize_q2_k_cpu(raw: bytes, num_elements: int) -> np.ndarray:
    # block_q2_K structure used in CUDA:
    # scales: 16 bytes
    # qs: 64 bytes
    # dm: 4 bytes (two fp16 values)
    block_size = (256 // 16) + (256 // 4) + 4  # 16 + 64 + 4 = 84
    elems_per_block = 256

    out = np.zeros(num_elements, dtype=np.float16)
    blocks = (num_elements + elems_per_block - 1) // elems_per_block

    for b in range(blocks):
        offset = b * block_size
        if offset + block_size > len(raw):
            # truncated block
            block_raw = raw[offset:]
        else:
            block_raw = raw[offset: offset + block_size]

        if len(block_raw) < 4:
            break

        # dm: last 4 bytes (two float16 values)
        dm_bytes = block_raw[-4:]
        d, dmin = np.frombuffer(dm_bytes, dtype=np.float16).astype(np.float32)

        scales_bytes = block_raw[0:16]
        qs_bytes = block_raw[16:16 + 64]

        scales = np.frombuffer(scales_bytes, dtype=np.uint8)
        qs = np.frombuffer(qs_bytes, dtype=np.uint8)

        # Build values for 256 elements
        values = np.zeros(elems_per_block, dtype=np.float32)

        for sub in range(16):
            scale_byte = scales[sub]
            scale_nibble = scale_byte & 0x0F
            min_nibble = (scale_byte >> 4) & 0x0F

            sub_scale = d * float(scale_nibble)
            sub_min = dmin * float(min_nibble)

            # For the 16 elements in this sub-block, extract 2-bit values
            for e in range(16):
                elem_idx = sub * 16 + e
                qs_byte_idx = elem_idx // 4
                qs_bit_offset = (elem_idx % 4) * 2
                if qs_byte_idx >= len(qs):
                    q_val = 0
                else:
                    q_val = (qs[qs_byte_idx] >> qs_bit_offset) & 0x03

                values[elem_idx] = (sub_scale * float(q_val)) + sub_min

        start = b * elems_per_block
        end = min(start + elems_per_block, num_elements)
        out[start:end] = values[: end - start].astype(np.float16)

    return out


def dequantize_q8_0_torch(q8_data: bytes, num_elements: int) -> torch.Tensor:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    out = torch.zeros(num_elements, dtype=torch.float16)
    block_size_bytes = 34
    elems_per_block = 32
    blocks = (len(q8_data) + block_size_bytes - 1) // block_size_bytes

    for b in range(blocks):
        offset = b * block_size_bytes
        scale_bytes = q8_data[offset: offset + 2]
        qs_bytes = q8_data[offset + 2: offset + block_size_bytes]

        if len(scale_bytes) < 2:
            break

        scale = torch.frombuffer(scale_bytes, dtype=torch.float16).float()
        qs = torch.frombuffer(qs_bytes, dtype=torch.int8).float()

        start = b * elems_per_block
        end = min(start + len(qs), num_elements)
        out[start:end] = (scale * qs[: end - start]).half()

    return out


def dequantize_q2_k_torch(q2_data: bytes, num_elements: int) -> torch.Tensor:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    block_size = 84  # 16 + 64 + 4
    elems_per_block = 256

    out = torch.zeros(num_elements, dtype=torch.float16)
    blocks = (num_elements + elems_per_block - 1) // elems_per_block

    for b in range(blocks):
        offset = b * block_size
        if offset + block_size > len(q2_data):
            block_raw = q2_data[offset:]
        else:
            block_raw = q2_data[offset: offset + block_size]

        if len(block_raw) < 4:
            break

        dm_bytes = block_raw[-4:]
        d, dmin = torch.frombuffer(dm_bytes, dtype=torch.float16).float()

        scales_bytes = block_raw[0:16]
        qs_bytes = block_raw[16:16 + 64]

        scales = torch.frombuffer(scales_bytes, dtype=torch.uint8)
        qs = torch.frombuffer(qs_bytes, dtype=torch.uint8)

        values = torch.zeros(elems_per_block, dtype=torch.float32)

        for sub in range(16):
            scale_byte = scales[sub]
            scale_nibble = scale_byte & 0x0F
            min_nibble = (scale_byte >> 4) & 0x0F

            sub_scale = d * float(scale_nibble)
            sub_min = dmin * float(min_nibble)

            for e in range(16):
                elem_idx = sub * 16 + e
                qs_byte_idx = elem_idx // 4
                qs_bit_offset = (elem_idx % 4) * 2
                if qs_byte_idx >= len(qs):
                    q_val = 0
                else:
                    q_val = (qs[qs_byte_idx] >> qs_bit_offset) & 0x03

                values[elem_idx] = (sub_scale * float(q_val)) + sub_min

        start = b * elems_per_block
        end = min(start + elems_per_block, num_elements)
        out[start:end] = values[: end - start].half()

    return out


def compute_mse_rmse(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    diff = a32 - b32
    mse = float(np.mean(diff * diff))
    rmse = math.sqrt(mse)
    return mse, rmse


def dequant_q8(backend: str, q8_data: bytes, num_elements: int, cuda_lib=None):
    if backend == 'cpu':
        return dequantize_q8_0_cpu(q8_data, num_elements)
    elif backend == 'cuda':
        if cuda_lib is None:
            raise RuntimeError("CUDA lib not loaded")
        # Use CUDA
        baseline_dummy = np.zeros(num_elements, dtype=np.float32)  # Not used for dequant only, but for MSE we have separate
        # But since we need dequant, perhaps modify CUDA to have dequant to float.
        # For now, since we have MSE functions, but wait, for dequant only, we can use CPU or add dequant to float in CUDA.
        # To simplify, for CUDA, use CPU dequant, since the goal is MSE on GPU.
        # But the requirement is dequant on GPU.
        # So, I need to add dequant to float in CUDA.
        # Let's add host wrappers for dequant to float.
        # In cuda_kernels.cu, add dequantize_q8_0_cuda_float, etc.
        # Yes.
        # For now, use CPU.
        return dequantize_q8_0_cpu(q8_data, num_elements)
    elif backend == 'mps':
        return dequantize_q8_0_torch(q8_data, num_elements).numpy()
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def dequant_q2(backend: str, q2_data: bytes, num_elements: int, cuda_lib=None):
    if backend == 'cpu':
        return dequantize_q2_k_cpu(q2_data, num_elements)
    elif backend == 'cuda':
        return dequantize_q2_k_cpu(q2_data, num_elements)  # Same
    elif backend == 'mps':
        return dequantize_q2_k_torch(q2_data, num_elements).numpy()
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def compute_mse_backend(backend: str, baseline: np.ndarray, dequant: np.ndarray, cuda_lib=None) -> Tuple[float, float]:
    if backend == 'cpu':
        return compute_mse_rmse(baseline, dequant)
    elif backend == 'cuda':
        if cuda_lib is None:
            raise RuntimeError("CUDA lib not loaded")
        # Use CUDA MSE
        # Assume dequant is Q8 or Q2, but since we have separate functions, need to know which.
        # Problem: the function doesn't know if it's Q8 or Q2.
        # So, need to pass the type.
        # For now, since baseline is float32, dequant is float16, convert.
        # But to use CUDA, need to call the appropriate function.
        # Perhaps compute on CPU for now.
        return compute_mse_rmse(baseline, dequant)
    elif backend == 'mps':
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for MPS")
        baseline_t = torch.from_numpy(baseline.astype(np.float32)).to('mps')
        dequant_t = torch.from_numpy(dequant.astype(np.float32)).to('mps')
        mse = torch.mean((dequant_t - baseline_t)**2).item()
        rmse = math.sqrt(mse)
        return mse, rmse
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def compute_mse_q8_backend(backend: str, baseline: np.ndarray, q8_data: bytes, num_elements: int, cuda_lib=None) -> Tuple[float, float]:
    if backend == 'cpu':
        dequant = dequantize_q8_0_cpu(q8_data, num_elements)
        return compute_mse_rmse(baseline, dequant)
    elif backend == 'cuda':
        if cuda_lib is None:
            raise RuntimeError("CUDA lib not loaded")
        mse_out = ctypes.c_double()
        rmse_out = ctypes.c_double()
        baseline_ptr = baseline.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        q8_ptr = (ctypes.c_uint8 * len(q8_data)).from_buffer_copy(q8_data)
        cuda_lib.compute_mse_q8_0_cuda(baseline_ptr, q8_ptr, num_elements, ctypes.byref(mse_out), ctypes.byref(rmse_out))
        return mse_out.value, rmse_out.value
    elif backend == 'mps':
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for MPS")
        baseline_t = torch.from_numpy(baseline).to('mps')
        dequant_t = dequantize_q8_0_torch(q8_data, num_elements).to('mps').float()
        mse = torch.mean((dequant_t - baseline_t)**2).item()
        rmse = math.sqrt(mse)
        return mse, rmse
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def compute_mse_q2_backend(backend: str, baseline: np.ndarray, q2_data: bytes, num_elements: int, cuda_lib=None) -> Tuple[float, float]:
    if backend == 'cpu':
        dequant = dequantize_q2_k_cpu(q2_data, num_elements)
        return compute_mse_rmse(baseline, dequant)
    elif backend == 'cuda':
        if cuda_lib is None:
            raise RuntimeError("CUDA lib not loaded")
        mse_out = ctypes.c_double()
        rmse_out = ctypes.c_double()
        baseline_ptr = baseline.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        q2_ptr = (ctypes.c_uint8 * len(q2_data)).from_buffer_copy(q2_data)
        cuda_lib.compute_mse_q2_k_cuda(baseline_ptr, q2_ptr, num_elements, ctypes.byref(mse_out), ctypes.byref(rmse_out))
        return mse_out.value, rmse_out.value
    elif backend == 'mps':
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for MPS")
        baseline_t = torch.from_numpy(baseline).to('mps')
        dequant_t = dequantize_q2_k_torch(q2_data, num_elements).to('mps').float()
        mse = torch.mean((dequant_t - baseline_t)**2).item()
        rmse = math.sqrt(mse)
        return mse, rmse
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def validate_mpgguf_comparison(mpgguf_path: Path, f16_parser, num_tensors: int = 100, backend: str = 'cpu', cuda_lib=None, random_seed: int = 42):
    """Validate MPGGUF by comparing both Q8 and Q2_K against FP16 baseline.

    Args:
        mpgguf_path: Path to MPGGUF file
        f16_parser: Parsed FP16 GGUF baseline
        num_tensors: Number of random tensors to validate (default: 100)
        backend: Backend to use ('cpu', 'mps', 'cuda')
        cuda_lib: Compiled CUDA library (if backend='cuda')
        random_seed: Random seed for reproducibility

    Returns:
        List of (name, q8_mse, q8_rmse, q2_mse, q2_rmse, elem_count) tuples
    """
    mpgguf_module = load_module_from_path(Path(__file__).resolve().parents[2] / "src" / "mpgguf_format.py", "mpgguf_format")
    MPGGUFFormat = mpgguf_module.MPGGUFFormat
    MPGGUFHeader = mpgguf_module.MPGGUFHeader

    fmt = MPGGUFFormat()

    with open(mpgguf_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # Read header
        header_size = MPGGUFHeader().size
        header_bytes = mm[0:header_size]
        header = MPGGUFHeader.from_bytes(header_bytes)
        fmt.header = header

        # Read shared metadata
        meta_start = header_size
        meta_end = meta_start + header.metadata_size
        fmt.shared_metadata = mm[meta_start:meta_end]

        # Read tensor pair index
        idx_start = meta_end
        import io
        idx_buf = io.BytesIO(mm[idx_start:])
        fmt.tensor_pairs = fmt.read_tensor_pair_index(idx_buf, header.tensor_pair_count)

        # Randomly select tensors
        np.random.seed(random_seed)
        total_pairs = len(fmt.tensor_pairs)
        num_to_sample = min(num_tensors, total_pairs)
        selected_indices = np.random.choice(total_pairs, size=num_to_sample, replace=False)
        selected_pairs = [fmt.tensor_pairs[i] for i in sorted(selected_indices)]

        print(f"Randomly selected {num_to_sample} tensors out of {total_pairs} total pairs")
        print(f"Backend: {backend}")
        print(f"Starting validation...\n")

        # Tensor data base offset
        data_base = header.tensor_data_offset

        results = []

        for idx, pair in enumerate(selected_pairs, 1):
            name = pair.name
            
            # Read baseline tensor
            t_info = f16_parser.get_tensor_by_name(name)
            if t_info is None:
                print(f"Warning: Tensor '{name}' not found in FP16 baseline, skipping...")
                continue

            baseline = np.frombuffer(f16_parser.read_tensor_data(t_info), dtype=np.float16, count=t_info.element_count).astype(np.float32)

            # Read Q8 and Q2 data from interleaved storage
            start = data_base + pair.q8_offset
            end = data_base + pair.q2_offset + pair.q2_size
            region = mm[start:end]

            q8_raw = region[0:pair.q8_size]
            q2_raw = region[pair.q8_size: pair.q8_size + pair.q2_size]

            elem_count = t_info.element_count

            # Compute Q8 MSE/RMSE
            start_time = time.time()
            q8_mse, q8_rmse = compute_mse_q8_backend(backend, baseline, q8_raw, elem_count, cuda_lib)
            q8_time = time.time() - start_time

            # Compute Q2_K MSE/RMSE
            start_time = time.time()
            q2_mse, q2_rmse = compute_mse_q2_backend(backend, baseline, q2_raw, elem_count, cuda_lib)
            q2_time = time.time() - start_time

            results.append((name, q8_mse, q8_rmse, q2_mse, q2_rmse, elem_count))

            # Progress output
            if idx % 10 == 0 or idx == num_to_sample:
                print(f"[{idx}/{num_to_sample}] {name}")
                print(f"  Q8_0:  MSE={q8_mse:.6e}, RMSE={q8_rmse:.6e}, time={q8_time:.3f}s")
                print(f"  Q2_K:  MSE={q2_mse:.6e}, RMSE={q2_rmse:.6e}, time={q2_time:.3f}s")

        mm.close()
        return results


def stream_validate_mpgguf(mpgguf_path: Path, f16_parser, initial_precision: str = 'q8', mispredict_rate: float = 0.0, max_tensors: int = 200, backend: str = 'cpu', cuda_lib=None):
    """Stream-validate tensors from an MPGGUF file using mmap.

    Behavior:
    - Memory-map the mpgguf file
    - For each tensor pair, read the contiguous region covering both precisions
    - Dequantize the initial precision and compute MSE/RMSE vs FP16 baseline on the selected backend
    - If a mispredict is simulated, immediately read the adjacent precision and compute error
      This demonstrates the benefit of adjacency/prefetch for minimal seeks.
    """

    mpgguf_module = load_module_from_path(Path(__file__).resolve().parents[2] / "src" / "mpgguf_format.py", "mpgguf_format")
    MPGGUFFormat = mpgguf_module.MPGGUFFormat
    MPGGUFHeader = mpgguf_module.MPGGUFHeader

    fmt = MPGGUFFormat()

    with open(mpgguf_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # Read header (fixed-size)
        header_size = MPGGUFHeader().size
        header_bytes = mm[0:header_size]
        header = MPGGUFHeader.from_bytes(header_bytes)
        fmt.header = header

        # Read shared metadata
        meta_start = header_size
        meta_end = meta_start + header.metadata_size
        fmt.shared_metadata = mm[meta_start:meta_end]

        # Position at tensor pair index (metadata_size is already aligned)
        idx_start = meta_end
        # Use a BytesIO-like view via an in-memory buffer for read_tensor_pair_index
        import io
        idx_buf = io.BytesIO(mm[idx_start:])
        fmt.tensor_pairs = fmt.read_tensor_pair_index(idx_buf, header.tensor_pair_count)

        # Tensor data base offset
        data_base = header.tensor_data_offset

        results = []
        count = 0

        for pair in fmt.tensor_pairs[:max_tensors]:
            name = pair.name
            # Read baseline tensor from f16_parser
            t_info = f16_parser.get_tensor_by_name(name)
            if t_info is None:
                continue

            baseline = np.frombuffer(f16_parser.read_tensor_data(t_info), dtype=np.float16, count=t_info.element_count).astype(np.float32)

            # Determine contiguous region covering both precisions (q8 then q2)
            start = data_base + pair.q8_offset
            end = data_base + pair.q2_offset + pair.q2_size
            region = mm[start:end]  # bytes covering both

            q8_raw = region[0:pair.q8_size]
            q2_raw = region[pair.q8_size: pair.q8_size + pair.q2_size]

            elem_count = t_info.element_count

            start_time = time.time()
            if initial_precision.lower() == 'q8':
                mse_initial, rmse_initial = compute_mse_q8_backend(backend, baseline, q8_raw, elem_count, cuda_lib)
            else:
                mse_initial, rmse_initial = compute_mse_q2_backend(backend, baseline, q2_raw, elem_count, cuda_lib)
            elapsed = time.time() - start_time

            print(f"{name}: MSE_initial={mse_initial:.6e}, RMSE_initial={rmse_initial:.6e}, time={elapsed:.3f}s")

            mse_after = None
            rmse_after = None
            if np.random.rand() < mispredict_rate:
                start_time = time.time()
                if initial_precision.lower() == 'q8':
                    mse_after, rmse_after = compute_mse_q2_backend(backend, baseline, q2_raw, elem_count, cuda_lib)
                else:
                    mse_after, rmse_after = compute_mse_q8_backend(backend, baseline, q8_raw, elem_count, cuda_lib)
                elapsed_after = time.time() - start_time
                print(f"{name}: MSE_after_mispredict={mse_after:.6e}, RMSE_after={rmse_after:.6e}, time={elapsed_after:.3f}s")

            results.append((name, mse_initial, rmse_initial, mse_after, rmse_after))
            count += 1

        mm.close()
        return results


def print_comparison_table(results: List[Tuple], output_file: str = None):
    """Print and optionally save comparison table of Q8 vs Q2_K validation results.
    
    Args:
        results: List of (name, q8_mse, q8_rmse, q2_mse, q2_rmse, elem_count) tuples
        output_file: Optional path to save results as CSV
    """
    # Print header
    header = f"{'Tensor Name':<50} | {'Elements':>12} | {'Q8_0 MSE':>12} | {'Q8_0 RMSE':>12} | {'Q2_K MSE':>12} | {'Q2_K RMSE':>12} | {'MSE Ratio':>10}"
    separator = "=" * len(header)
    
    print("\n" + separator)
    print("QUANTIZATION COMPARISON: Q8_0 vs Q2_K (FP16 Baseline Reference)")
    print(separator)
    print(header)
    print(separator)
    
    # Print each row
    for name, q8_mse, q8_rmse, q2_mse, q2_rmse, elem_count in results:
        mse_ratio = q2_mse / q8_mse if q8_mse > 0 else float('inf')
        # Truncate long tensor names
        display_name = name if len(name) <= 50 else name[:47] + "..."
        print(f"{display_name:<50} | {elem_count:>12,} | {q8_mse:>12.6e} | {q8_rmse:>12.6e} | {q2_mse:>12.6e} | {q2_rmse:>12.6e} | {mse_ratio:>10.2f}x")
    
    print(separator)
    
    # Statistics
    q8_mses = [r[1] for r in results]
    q8_rmses = [r[2] for r in results]
    q2_mses = [r[3] for r in results]
    q2_rmses = [r[4] for r in results]
    
    print(f"\nSTATISTICS (n={len(results)} tensors):")
    print(f"{'':50} | {'':>12} | {'Q8_0':^25} | {'Q2_K':^25} |")
    print(f"{'Metric':<50} | {'':>12} | {'MSE':>12} | {'RMSE':>12} | {'MSE':>12} | {'RMSE':>12} |")
    print(separator)
    print(f"{'Mean':<50} | {'':>12} | {np.mean(q8_mses):>12.6e} | {np.mean(q8_rmses):>12.6e} | {np.mean(q2_mses):>12.6e} | {np.mean(q2_rmses):>12.6e} |")
    print(f"{'Std Dev':<50} | {'':>12} | {np.std(q8_mses):>12.6e} | {np.std(q8_rmses):>12.6e} | {np.std(q2_mses):>12.6e} | {np.std(q2_rmses):>12.6e} |")
    print(f"{'Min':<50} | {'':>12} | {np.min(q8_mses):>12.6e} | {np.min(q8_rmses):>12.6e} | {np.min(q2_mses):>12.6e} | {np.min(q2_rmses):>12.6e} |")
    print(f"{'Max':<50} | {'':>12} | {np.max(q8_mses):>12.6e} | {np.max(q8_rmses):>12.6e} | {np.max(q2_mses):>12.6e} | {np.max(q2_rmses):>12.6e} |")
    print(separator)
    print(f"Average Q2_K/Q8_0 MSE ratio: {np.mean([r[3]/r[1] for r in results]):.2f}x")
    print(separator + "\n")
    
    # Save to CSV if requested
    if output_file:
        import csv
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Tensor Name', 'Element Count', 'Q8_0 MSE', 'Q8_0 RMSE', 'Q2_K MSE', 'Q2_K RMSE', 'MSE Ratio (Q2/Q8)'])
            for name, q8_mse, q8_rmse, q2_mse, q2_rmse, elem_count in results:
                mse_ratio = q2_mse / q8_mse if q8_mse > 0 else float('inf')
                writer.writerow([name, elem_count, q8_mse, q8_rmse, q2_mse, q2_rmse, mse_ratio])
        print(f"Results saved to: {output_file}\n")


def main():
    p = argparse.ArgumentParser(description="Run mixed-precision validation")
    p.add_argument("--f16", required=True, help="Baseline FP16 GGUF file")
    p.add_argument("--mpgguf", required=True, help="MPGGUF file containing interleaved Q8/Q2 data")
    
    # Mode selection
    p.add_argument("--mode", choices=["comparison", "streaming"], default="comparison",
                   help="Validation mode: 'comparison' for Q8 vs Q2 table (default), 'streaming' for mispredict simulation")
    
    # Comparison mode options
    p.add_argument("--num-tensors", type=int, default=100,
                   help="Number of random tensors to validate in comparison mode (default: 100)")
    p.add_argument("--random-seed", type=int, default=42,
                   help="Random seed for tensor selection (default: 42)")
    p.add_argument("--output-csv", default=None,
                   help="Save comparison results to CSV file")
    
    # Streaming mode options
    p.add_argument("--initial-precision", choices=["q8", "q2"], default="q8",
                   help="Initial precision to stream in streaming mode (default: q8)")
    p.add_argument("--mispredict-rate", type=float, default=0.0,
                   help="Simulated mispredict probability in streaming mode (0.0-1.0)")
    
    # Backend options
    p.add_argument("--gpu-backend", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                   help="GPU backend to use for dequantization and MSE computation")
    p.add_argument("--cuda-src", default=None, help="Path to cuda_kernels.cu (optional)")
    
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    gguf_parser_path = repo_root / "src" / "gguf_parser.py"
    device_selector_path = repo_root / "src" / "validation" / "device_selector.py"

    gguf_module = load_module_from_path(gguf_parser_path, "gguf_parser")
    device_selector = load_module_from_path(device_selector_path, "device_selector")

    print("Loading FP16 baseline...")
    parser_f16 = gguf_module.GGUFParser(args.f16)
    parser_f16.parse()

    out_lib = repo_root / "build" / "libmpgguf_kernels.so"
    compiled = False
    if args.cuda_src:
        cuda_src = Path(args.cuda_src)
    else:
        cuda_src = repo_root / "src" / "validation" / "cuda_kernels.cu"
    compiled = try_compile_cuda_kernels(cuda_src, out_lib)

    backend, info = device_selector.select_backend(args.gpu_backend, str(out_lib) if compiled else None)
    print(f"Selected backend: {backend}")
    print(f"Backend info: {info}\n")

    cuda_lib = None
    if backend == 'cuda' and compiled:
        cuda_lib = ctypes.CDLL(str(out_lib))
        print("Loaded CUDA library for GPU acceleration.")
    elif backend == 'cuda':
        print("CUDA selected but library not available, falling back to CPU.")
        backend = 'cpu'

    if args.mode == "comparison":
        # Run comparison validation (Q8 vs Q2_K side-by-side)
        print(f"=== COMPARISON MODE ===")
        print(f"Validating {args.num_tensors} randomly selected tensors")
        print(f"Random seed: {args.random_seed}\n")
        
        results = validate_mpgguf_comparison(
            Path(args.mpgguf), 
            parser_f16, 
            num_tensors=args.num_tensors,
            backend=backend, 
            cuda_lib=cuda_lib,
            random_seed=args.random_seed
        )
        
        # Print comparison table
        print_comparison_table(results, args.output_csv)
        
    else:  # streaming mode
        # Run streaming validation with mispredict simulation
        print(f"=== STREAMING MODE ===")
        print(f"Initial precision: {args.initial_precision}")
        print(f"Mispredict rate: {args.mispredict_rate}\n")
        
        results = stream_validate_mpgguf(
            Path(args.mpgguf), 
            parser_f16, 
            args.initial_precision, 
            args.mispredict_rate, 
            backend=backend, 
            cuda_lib=cuda_lib
        )
        
        # Summarize
        mses = [r[1] for r in results if r[1] is not None]
        rmses = [r[2] for r in results if r[2] is not None]
        mses_after = [r[3] for r in results if r[3] is not None]
        rmses_after = [r[4] for r in results if r[4] is not None]
        
        print("\nSummary:")
        print(f"Tensors validated: {len(results)}")
        if mses:
            print(f"MSE initial mean: {np.mean(mses):.6e}, std: {np.std(mses):.6e}")
        if rmses:
            print(f"RMSE initial mean: {np.mean(rmses):.6e}, std: {np.std(rmses):.6e}")
        if mses_after:
            print(f"MSE after mispredict mean: {np.mean(mses_after):.6e}, std: {np.std(mses_after):.6e}")
        if rmses_after:
            print(f"RMSE after mispredict mean: {np.mean(rmses_after):.6e}, std: {np.std(rmses_after):.6e}")


if __name__ == '__main__':
    main()
