"""Backend selection helper for MPGGUF validation

Selects backend in order: CUDA -> MPS -> CPU.
Provides small wrapper so validation script can uniformly use the selected backend.
"""
from typing import List, Optional, Tuple


def get_available_backends() -> List[str]:
    """Get list of available backends in order of preference."""
    backends = []
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            backends.append('cuda')
    except Exception:
        pass
    
    # Check MPS
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            backends.append('mps')
    except Exception:
        pass
    
    # CPU always available
    backends.append('cpu')
    
    return backends


def select_backend(preferred: str = 'auto', kernel_lib_path: Optional[str] = None) -> Tuple[str, dict]:
    """Select execution backend.

    Args:
        preferred: 'auto', 'cuda', 'mps', or 'cpu'
        kernel_lib_path: Path to compiled CUDA kernel library (for CUDA backend)

    Returns:
        (backend, info) where backend is one of 'cuda', 'mps', 'cpu' and info is an environment dict.
    """
    available = get_available_backends()
    info = {'available_backends': available}
    
    if preferred == 'auto':
        # Try in order: cuda, mps, cpu
        for backend in ['cuda', 'mps', 'cpu']:
            if backend in available:
                selected = backend
                break
    else:
        if preferred in available:
            selected = preferred
        else:
            # Fallback to first available
            selected = available[0] if available else 'cpu'
    
    info['selected'] = selected
    
    # Additional info for selected backend
    if selected == 'cuda':
        try:
            import torch
            info['cuda_devices'] = torch.cuda.device_count()
            if kernel_lib_path:
                import os
                info['kernel_lib'] = os.path.abspath(kernel_lib_path) if os.path.exists(kernel_lib_path) else None
        except Exception:
            pass
    elif selected == 'mps':
        info['mps'] = True
    
    return selected, info


# Backward compatibility
def select_device(kernel_lib_path: Optional[str] = None) -> Tuple[str, dict]:
    return select_backend('auto', kernel_lib_path)
