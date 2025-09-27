#!/usr/bin/env python3
"""Test script to verify MPGGUF builder quantization type support."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from gguf_parser import GGMLType
from mpgguf_builder import MPGGUFBuilder

def test_quantization_support():
    """Test that builder supports various 2-bit quantization types."""
    
    print("Testing MPGGUF Builder Quantization Type Support")
    print("=" * 50)
    
    # Mock tensor info for testing
    class MockTensor:
        def __init__(self, name, ggml_type):
            self.name = name
            self.ggml_type = ggml_type
    
    # Create mock builder instance
    builder = MPGGUFBuilder("dummy_q8.gguf", "dummy_q2.gguf", "dummy_out.mpgguf")
    
    # Test Q8_0 tensor
    q8_tensor = MockTensor("test.weight", GGMLType.Q8_0)
    
    # Test various 2-bit types
    test_cases = [
        ("Q2_K", GGMLType.Q2_K),
        ("IQ2_XXS", GGMLType.IQ2_XXS), 
        ("IQ2_XS", GGMLType.IQ2_XS),
        ("Q2_0", GGMLType.Q2_0),
        ("Q4_0 (should fail)", GGMLType.Q4_0),  # This should fail
        ("F32 (should fail)", GGMLType.F32),   # This should fail
    ]
    
    for type_name, ggml_type in test_cases:
        q2_tensor = MockTensor("test.weight", ggml_type)
        is_valid = builder._is_quantized_tensor(q8_tensor, q2_tensor)
        
        status = "✅ SUPPORTED" if is_valid else "❌ NOT SUPPORTED"
        print(f"{type_name:15} -> {status}")
    
    print("\n" + "=" * 50)
    print("Quantization type validation complete!")

if __name__ == "__main__":
    test_quantization_support()