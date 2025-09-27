#!/usr/bin/env python3
"""
Mixed-Precision GGUF (MPGGUF) Format Specification

Defines the binary format for storing both Q8_0 and Q2_0 quantizations
of the same model in a single file with interleaved tensor packing.
"""

import struct
from typing import Dict, List, Tuple, Any, BinaryIO
from dataclasses import dataclass
from enum import IntEnum
from gguf_parser import GGUFParser, TensorInfo, GGMLType


class MPGGUFVersion(IntEnum):
    """MPGGUF format versions"""
    V1 = 1


@dataclass
class MPGGUFTensorPair:
    """Information about a paired tensor in MPGGUF"""
    name: str
    shape: Tuple[int, ...]
    q8_offset: int
    q8_size: int
    q2_offset: int  
    q2_size: int
    
    @property
    def total_size(self) -> int:
        return self.q8_size + self.q2_size


@dataclass
class MPGGUFHeader:
    """MPGGUF file header"""
    magic: bytes = b'MPGG'  # Mixed-Precision GGUF magic
    version: int = MPGGUFVersion.V1
    base_gguf_version: int = 3
    tensor_pair_count: int = 0
    metadata_size: int = 0
    tensor_data_offset: int = 0
    
    def to_bytes(self) -> bytes:
        """Serialize header to bytes"""
        return struct.pack('<4sIIQQQ', 
                          self.magic,
                          self.version,
                          self.base_gguf_version,
                          self.tensor_pair_count,
                          self.metadata_size,
                          self.tensor_data_offset)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'MPGGUFHeader':
        """Deserialize header from bytes"""
        magic, version, base_version, tensor_count, metadata_size, data_offset = \
            struct.unpack('<4sIIQQQ', data)
        
        if magic != b'MPGG':
            raise ValueError(f"Invalid MPGGUF magic: {magic}")
        
        return cls(magic, version, base_version, tensor_count, metadata_size, data_offset)
    
    @property
    def size(self) -> int:
        """Size of the header in bytes"""
        return 36  # 4 + 4 + 4 + 8 + 8 + 8 = 36 bytes


class MPGGUFFormat:
    """
    MPGGUF Binary Format Layout:
    
    1. MPGGUF Header (32 bytes, aligned)
       - Magic: 'MPGG' (4 bytes)
       - Version: uint32 (4 bytes)
       - Base GGUF Version: uint32 (4 bytes)
       - Tensor Pair Count: uint64 (8 bytes)
       - Metadata Size: uint64 (8 bytes)
       - Tensor Data Offset: uint64 (8 bytes)
    
    2. Shared Metadata (variable size, from base GGUF)
       - Architecture info, tokenizer, etc.
       - Everything except tensor-specific quantization info
    
    3. Tensor Pair Index (aligned to 32-byte boundary)
       - Array of tensor pair descriptors
       - Each entry: name, shape, Q8 offset/size, Q2 offset/size
    
    4. Interleaved Tensor Data (aligned to 32-byte boundary)
       - For each tensor: Q8_0 data immediately followed by Q2_0 data
       - Padding between tensor pairs for alignment
    
    Advantages of this layout:
    - Minimal seeks: Q8 and Q2 versions of same tensor are adjacent
    - Cache-friendly: fetching one precision pre-loads the other
    - Shared metadata: no duplication of model architecture info
    - Aligned access: all major sections on 32-byte boundaries
    """
    
    MAGIC = b'MPGG'
    CURRENT_VERSION = MPGGUFVersion.V1
    ALIGNMENT = 32
    
    def __init__(self):
        self.header = MPGGUFHeader()
        self.shared_metadata: bytes = b''
        self.tensor_pairs: List[MPGGUFTensorPair] = []
    
    @staticmethod
    def align_size(size: int, alignment: int = None) -> int:
        """Align size to boundary"""
        if alignment is None:
            alignment = MPGGUFFormat.ALIGNMENT
        return (size + alignment - 1) // alignment * alignment
    
    @staticmethod
    def calculate_interleaved_offsets(tensor_pairs: List[Tuple[str, TensorInfo, TensorInfo]]) -> List[MPGGUFTensorPair]:
        """
        Calculate offsets for interleaved tensor packing.
        
        Layout for each tensor pair:
        [Q8_0 data][Q2_0 data][padding to alignment]
        
        Args:
            tensor_pairs: List of (name, q8_tensor, q2_tensor) tuples
            
        Returns:
            List of MPGGUFTensorPair with calculated offsets
        """
        result = []
        current_offset = 0
        
        for name, q8_tensor, q2_tensor in tensor_pairs:
            # Ensure shapes match
            if q8_tensor.shape != q2_tensor.shape:
                raise ValueError(f"Tensor shape mismatch for {name}: "
                               f"Q8={q8_tensor.shape} vs Q2={q2_tensor.shape}")
            
            # Q8_0 data comes first
            q8_offset = current_offset
            q8_size = q8_tensor.size
            
            # Q2_0 data immediately follows
            q2_offset = q8_offset + q8_size
            q2_size = q2_tensor.size
            
            # Create tensor pair info
            pair = MPGGUFTensorPair(
                name=name,
                shape=q8_tensor.shape,
                q8_offset=q8_offset,
                q8_size=q8_size,
                q2_offset=q2_offset,
                q2_size=q2_size
            )
            
            result.append(pair)
            
            # Move to next aligned position
            total_pair_size = q8_size + q2_size
            current_offset = q2_offset + q2_size
            current_offset = MPGGUFFormat.align_size(current_offset)
        
        return result
    
    def create_shared_metadata(self, q8_parser: GGUFParser, q2_parser: GGUFParser) -> bytes:
        """
        Create shared metadata section by merging compatible metadata from both files.
        
        Includes:
        - Architecture information
        - Tokenizer data
        - Model hyperparameters
        - Everything except quantization-specific keys
        
        Args:
            q8_parser: Parser for Q8_0 GGUF file
            q2_parser: Parser for Q2_0 GGUF file
            
        Returns:
            Serialized metadata bytes
        """
        # Start with Q8 metadata as base
        shared_metadata = q8_parser.metadata.copy()
        
        # Remove quantization-specific keys
        quantization_keys = [
            'general.file_type',
            'quantization_version'
        ]
        
        for key in quantization_keys:
            shared_metadata.pop(key, None)
        
        # Add MPGGUF-specific metadata
        shared_metadata['general.file_type'] = 'MPGGUF'
        shared_metadata['mpgguf.version'] = self.CURRENT_VERSION
        shared_metadata['mpgguf.q8_file_type'] = q8_parser.get_file_type()
        shared_metadata['mpgguf.q2_file_type'] = q2_parser.get_file_type()
        
        # Validate compatibility
        self._validate_compatibility(q8_parser.metadata, q2_parser.metadata)
        
        # Serialize metadata (simplified - in practice would use GGUF format)
        import json
        metadata_json = json.dumps(shared_metadata, indent=2)
        return metadata_json.encode('utf-8')
    
    def _validate_compatibility(self, q8_metadata: Dict[str, Any], q2_metadata: Dict[str, Any]) -> None:
        """Validate that two GGUF files are compatible for merging"""
        
        # Keys that must match exactly
        required_match_keys = [
            'general.architecture',
            'general.name',
        ]
        
        # Keys that should match (architecture-specific)
        arch = q8_metadata.get('general.architecture', '')
        arch_keys = [
            f'{arch}.vocab_size',
            f'{arch}.context_length', 
            f'{arch}.embedding_length',
            f'{arch}.block_count',
            f'{arch}.attention.head_count',
        ]
        
        all_match_keys = required_match_keys + arch_keys
        
        for key in all_match_keys:
            if key in q8_metadata and key in q2_metadata:
                if q8_metadata[key] != q2_metadata[key]:
                    raise ValueError(f"Metadata mismatch for key '{key}': "
                                   f"Q8={q8_metadata[key]} vs Q2={q2_metadata[key]}")
    
    def write_tensor_pair_index(self, f: BinaryIO, tensor_pairs: List[MPGGUFTensorPair]) -> None:
        """Write tensor pair index to file"""
        for pair in tensor_pairs:
            # Write tensor name
            name_bytes = pair.name.encode('utf-8')
            f.write(struct.pack('<Q', len(name_bytes)))
            f.write(name_bytes)
            
            # Write shape
            f.write(struct.pack('<I', len(pair.shape)))
            for dim in pair.shape:
                f.write(struct.pack('<Q', dim))
            
            # Write offsets and sizes
            f.write(struct.pack('<QQQQ', 
                              pair.q8_offset, pair.q8_size,
                              pair.q2_offset, pair.q2_size))
    
    def read_tensor_pair_index(self, f: BinaryIO, count: int) -> List[MPGGUFTensorPair]:
        """Read tensor pair index from file"""
        tensor_pairs = []
        
        for _ in range(count):
            # Read tensor name
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')
            
            # Read shape
            n_dims = struct.unpack('<I', f.read(4))[0]
            shape = []
            for _ in range(n_dims):
                dim = struct.unpack('<Q', f.read(8))[0]
                shape.append(dim)
            
            # Read offsets and sizes
            q8_offset, q8_size, q2_offset, q2_size = struct.unpack('<QQQQ', f.read(32))
            
            pair = MPGGUFTensorPair(
                name=name,
                shape=tuple(shape),
                q8_offset=q8_offset,
                q8_size=q8_size,
                q2_offset=q2_offset,
                q2_size=q2_size
            )
            
            tensor_pairs.append(pair)
        
        return tensor_pairs
    
    def get_precision_data(self, f: BinaryIO, tensor_name: str, precision: str) -> bytes:
        """
        Get tensor data for specified precision.
        
        Args:
            f: File handle positioned at tensor data section
            tensor_name: Name of the tensor
            precision: 'q8' or 'q2'
            
        Returns:
            Raw tensor data bytes
        """
        # Find tensor pair
        pair = None
        for tp in self.tensor_pairs:
            if tp.name == tensor_name:
                pair = tp
                break
        
        if pair is None:
            raise ValueError(f"Tensor '{tensor_name}' not found")
        
        # Seek to appropriate offset and read data
        if precision.lower() == 'q8':
            f.seek(self.header.tensor_data_offset + pair.q8_offset)
            return f.read(pair.q8_size)
        elif precision.lower() == 'q2':
            f.seek(self.header.tensor_data_offset + pair.q2_offset)
            return f.read(pair.q2_size)
        else:
            raise ValueError(f"Invalid precision: {precision}. Must be 'q8' or 'q2'")


if __name__ == "__main__":
    # Example usage and testing
    print("MPGGUF Format Specification")
    print(f"Current Version: {MPGGUFFormat.CURRENT_VERSION}")
    print(f"Header Size: {MPGGUFHeader().size} bytes")
    print(f"Alignment: {MPGGUFFormat.ALIGNMENT} bytes")