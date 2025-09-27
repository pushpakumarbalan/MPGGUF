#!/usr/bin/env python3
"""
GGUF File Format Parser

Parses GGUF files to extract metadata, tensor information, and data.
Based on the GGUF specification: https://deepwiki.com/ggml-org/llama.cpp/6.1-gguf-file-format
"""

import struct
import mmap
from typing import Dict, List, Tuple, Any, Optional, BinaryIO
from dataclasses import dataclass
from enum import IntEnum
import numpy as np


class GGUFValueType(IntEnum):
    """GGUF metadata value types"""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLType(IntEnum):
    """GGML tensor data types"""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    Q2_0 = 18  # Custom Q2_0 type


@dataclass
class TensorInfo:
    """Information about a tensor in the GGUF file"""
    name: str
    n_dims: int
    dimensions: List[int]
    ggml_type: GGMLType
    offset: int
    size: int
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.dimensions)
    
    @property
    def element_count(self) -> int:
        return int(np.prod(self.dimensions))


@dataclass
class GGUFHeader:
    """GGUF file header"""
    magic: bytes
    version: int
    tensor_count: int
    metadata_kv_count: int


class GGUFParser:
    """Parser for GGUF files"""
    
    MAGIC = b'GGUF'
    SUPPORTED_VERSIONS = [3]
    
    # Type size mappings
    TYPE_SIZES = {
        GGUFValueType.UINT8: 1,
        GGUFValueType.INT8: 1,
        GGUFValueType.UINT16: 2,
        GGUFValueType.INT16: 2,
        GGUFValueType.UINT32: 4,
        GGUFValueType.INT32: 4,
        GGUFValueType.FLOAT32: 4,
        GGUFValueType.BOOL: 1,
        GGUFValueType.UINT64: 8,
        GGUFValueType.INT64: 8,
        GGUFValueType.FLOAT64: 8,
    }
    
    # GGML type sizes (in bytes per element)
    GGML_TYPE_SIZES = {
        GGMLType.F32: 4,
        GGMLType.F16: 2,
        GGMLType.Q4_0: 34,  # Block size for Q4_0
        GGMLType.Q4_1: 36,  # Block size for Q4_1
        GGMLType.Q5_0: 34,  # Block size for Q5_0
        GGMLType.Q5_1: 36,  # Block size for Q5_1
        GGMLType.Q8_0: 34,  # Block size for Q8_0: 2 bytes (scale) + 32 bytes (data)
        GGMLType.Q8_1: 36,  # Block size for Q8_1
        GGMLType.Q2_0: 18,  # Block size for Q2_0: 2 bytes (scale) + 16 bytes (data)
    }
    
    def __init__(self, filename: str):
        self.filename = filename
        self.header: Optional[GGUFHeader] = None
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[TensorInfo] = []
        self.tensor_data_offset: int = 0
        
    def parse(self) -> None:
        """Parse the entire GGUF file"""
        with open(self.filename, 'rb') as f:
            # Parse header
            self.header = self._parse_header(f)
            
            # Parse metadata
            self.metadata = self._parse_metadata(f)
            
            # Parse tensor info
            self.tensors = self._parse_tensor_info(f)
            
            # Calculate tensor data offset (aligned to 32-byte boundary)
            current_pos = f.tell()
            self.tensor_data_offset = self._align_offset(current_pos, 32)
    
    def _parse_header(self, f: BinaryIO) -> GGUFHeader:
        """Parse GGUF header"""
        magic = f.read(4)
        if magic != self.MAGIC:
            raise ValueError(f"Invalid GGUF magic: {magic}")
        
        version = struct.unpack('<I', f.read(4))[0]
        if version not in self.SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported GGUF version: {version}")
        
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
        
        return GGUFHeader(magic, version, tensor_count, metadata_kv_count)
    
    def _parse_metadata(self, f: BinaryIO) -> Dict[str, Any]:
        """Parse metadata key-value pairs"""
        metadata = {}
        
        for _ in range(self.header.metadata_kv_count):
            # Read key
            key = self._read_string(f)
            
            # Read value type
            value_type = GGUFValueType(struct.unpack('<I', f.read(4))[0])
            
            # Read value
            value = self._read_value(f, value_type)
            
            metadata[key] = value
        
        return metadata
    
    def _parse_tensor_info(self, f: BinaryIO) -> List[TensorInfo]:
        """Parse tensor information"""
        tensors = []
        current_offset = 0
        
        for _ in range(self.header.tensor_count):
            # Read tensor name
            name = self._read_string(f)
            
            # Read dimensions
            n_dims = struct.unpack('<I', f.read(4))[0]
            dimensions = []
            for _ in range(n_dims):
                dim = struct.unpack('<Q', f.read(8))[0]
                dimensions.append(dim)
            
            # Read tensor type
            ggml_type = GGMLType(struct.unpack('<I', f.read(4))[0])
            
            # Read tensor offset
            offset = struct.unpack('<Q', f.read(8))[0]
            
            # Calculate tensor size
            size = self._calculate_tensor_size(dimensions, ggml_type)
            
            tensor_info = TensorInfo(
                name=name,
                n_dims=n_dims,
                dimensions=dimensions,
                ggml_type=ggml_type,
                offset=offset,
                size=size
            )
            
            tensors.append(tensor_info)
        
        return tensors
    
    def _read_string(self, f: BinaryIO) -> str:
        """Read a string from the file"""
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')
    
    def _read_value(self, f: BinaryIO, value_type: GGUFValueType) -> Any:
        """Read a value based on its type"""
        if value_type == GGUFValueType.STRING:
            return self._read_string(f)
        elif value_type == GGUFValueType.ARRAY:
            # Read array type and length
            array_type = GGUFValueType(struct.unpack('<I', f.read(4))[0])
            array_length = struct.unpack('<Q', f.read(8))[0]
            
            # Read array elements
            array_values = []
            for _ in range(array_length):
                value = self._read_value(f, array_type)
                array_values.append(value)
            
            return array_values
        else:
            # Read scalar value
            size = self.TYPE_SIZES[value_type]
            data = f.read(size)
            
            if value_type == GGUFValueType.UINT8:
                return struct.unpack('<B', data)[0]
            elif value_type == GGUFValueType.INT8:
                return struct.unpack('<b', data)[0]
            elif value_type == GGUFValueType.UINT16:
                return struct.unpack('<H', data)[0]
            elif value_type == GGUFValueType.INT16:
                return struct.unpack('<h', data)[0]
            elif value_type == GGUFValueType.UINT32:
                return struct.unpack('<I', data)[0]
            elif value_type == GGUFValueType.INT32:
                return struct.unpack('<i', data)[0]
            elif value_type == GGUFValueType.FLOAT32:
                return struct.unpack('<f', data)[0]
            elif value_type == GGUFValueType.BOOL:
                return bool(struct.unpack('<B', data)[0])
            elif value_type == GGUFValueType.UINT64:
                return struct.unpack('<Q', data)[0]
            elif value_type == GGUFValueType.INT64:
                return struct.unpack('<q', data)[0]
            elif value_type == GGUFValueType.FLOAT64:
                return struct.unpack('<d', data)[0]
            else:
                raise ValueError(f"Unknown value type: {value_type}")
    
    def _calculate_tensor_size(self, dimensions: List[int], ggml_type: GGMLType) -> int:
        """Calculate the size of a tensor in bytes"""
        element_count = int(np.prod(dimensions))
        
        if ggml_type in [GGMLType.F32, GGMLType.F16]:
            # Simple types: size = elements * type_size
            return element_count * self.GGML_TYPE_SIZES[ggml_type]
        else:
            # Quantized types: calculate based on block structure
            if ggml_type == GGMLType.Q8_0:
                # Q8_0: 32 elements per block, 34 bytes per block
                blocks = (element_count + 31) // 32
                return blocks * 34
            elif ggml_type == GGMLType.Q2_0:
                # Q2_0: 32 elements per block, 18 bytes per block (custom)
                blocks = (element_count + 31) // 32
                return blocks * 18
            else:
                # Default block-based calculation
                block_size = self.GGML_TYPE_SIZES.get(ggml_type, 32)
                blocks = (element_count + 31) // 32
                return blocks * block_size
    
    def _align_offset(self, offset: int, alignment: int) -> int:
        """Align offset to specified boundary"""
        return (offset + alignment - 1) // alignment * alignment
    
    def read_tensor_data(self, tensor: TensorInfo) -> bytes:
        """Read raw tensor data from the file"""
        with open(self.filename, 'rb') as f:
            f.seek(self.tensor_data_offset + tensor.offset)
            return f.read(tensor.size)
    
    def get_tensor_by_name(self, name: str) -> Optional[TensorInfo]:
        """Get tensor info by name"""
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        return None
    
    def get_architecture(self) -> str:
        """Get model architecture from metadata"""
        return self.metadata.get('general.architecture', 'unknown')
    
    def get_file_type(self) -> str:
        """Get file quantization type"""
        file_type = self.metadata.get('general.file_type', 0)
        type_names = {
            0: 'F32',
            1: 'F16',
            2: 'Q4_0',
            3: 'Q4_1',
            8: 'Q8_0',
            18: 'Q2_0',
        }
        return type_names.get(file_type, f'UNKNOWN_{file_type}')
    
    def print_summary(self) -> None:
        """Print a summary of the GGUF file"""
        print(f"GGUF File: {self.filename}")
        print(f"Version: {self.header.version}")
        print(f"Architecture: {self.get_architecture()}")
        print(f"File Type: {self.get_file_type()}")
        print(f"Tensors: {len(self.tensors)}")
        print(f"Metadata Keys: {len(self.metadata)}")
        print(f"Tensor Data Offset: 0x{self.tensor_data_offset:08x}")
        
        print("\nTensor Summary:")
        for tensor in self.tensors[:10]:  # Show first 10 tensors
            print(f"  {tensor.name}: {tensor.shape} ({tensor.ggml_type.name})")
        
        if len(self.tensors) > 10:
            print(f"  ... and {len(self.tensors) - 10} more tensors")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python gguf_parser.py <gguf_file>")
        sys.exit(1)
    
    parser = GGUFParser(sys.argv[1])
    parser.parse()
    parser.print_summary()