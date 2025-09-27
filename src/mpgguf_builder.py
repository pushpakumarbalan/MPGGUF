#!/usr/bin/env python3
"""
MPGGUF Builder - Main script for merging Q8_0 and Q2_0 GGUF files

Usage:
    python mpgguf_builder.py --q8 model.Q8_0.gguf --q2 model.Q2_0.gguf --out model.mpgguf
"""

import os
import sys
import argparse
import logging
from typing import List, Tuple, BinaryIO
from pathlib import Path

# Import our modules
from gguf_parser import GGUFParser, TensorInfo
from mpgguf_format import MPGGUFFormat, MPGGUFHeader, MPGGUFTensorPair


class MPGGUFBuilder:
    """Main class for building MPGGUF files from Q8_0 and Q2_0 GGUF files"""
    
    def __init__(self, q8_path: str, q2_path: str, output_path: str, verbose: bool = False):
        self.q8_path = Path(q8_path)
        self.q2_path = Path(q2_path)
        self.output_path = Path(output_path)
        self.verbose = verbose
        
        # Setup logging
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize parsers
        self.q8_parser = None
        self.q2_parser = None
        self.format = MPGGUFFormat()
        
    def validate_inputs(self) -> None:
        """Validate input files exist and are accessible"""
        if not self.q8_path.exists():
            raise FileNotFoundError(f"Q8_0 GGUF file not found: {self.q8_path}")
        
        if not self.q2_path.exists():
            raise FileNotFoundError(f"Q2_0 GGUF file not found: {self.q2_path}")
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Input Q8_0 file: {self.q8_path} ({self.q8_path.stat().st_size:,} bytes)")
        self.logger.info(f"Input Q2_0 file: {self.q2_path} ({self.q2_path.stat().st_size:,} bytes)")
        self.logger.info(f"Output MPGGUF file: {self.output_path}")
    
    def parse_input_files(self) -> None:
        """Parse both input GGUF files"""
        self.logger.info("Parsing Q8_0 GGUF file...")
        self.q8_parser = GGUFParser(str(self.q8_path))
        self.q8_parser.parse()
        
        self.logger.info("Parsing Q2_0 GGUF file...")
        self.q2_parser = GGUFParser(str(self.q2_path))
        self.q2_parser.parse()
        
        # Log summaries
        if self.verbose:
            print("\n=== Q8_0 GGUF Summary ===")
            self.q8_parser.print_summary()
            print("\n=== Q2_0 GGUF Summary ===")
            self.q2_parser.print_summary()
            print()
    
    def match_tensors(self) -> List[Tuple[str, TensorInfo, TensorInfo]]:
        """
        Match tensors between Q8_0 and Q2_0 files.
        
        Returns:
            List of (tensor_name, q8_tensor, q2_tensor) tuples for quantized tensors
        """
        self.logger.info("Matching tensors between Q8_0 and Q2_0 files...")
        
        # Create tensor name mappings
        q8_tensors = {tensor.name: tensor for tensor in self.q8_parser.tensors}
        q2_tensors = {tensor.name: tensor for tensor in self.q2_parser.tensors}
        
        # Find common quantized tensors
        matched_pairs = []
        unquantized_tensors = []
        
        for name, q8_tensor in q8_tensors.items():
            if name in q2_tensors:
                q2_tensor = q2_tensors[name]
                
                # Check if this is a quantized tensor (not embedding/output layers)
                if self._is_quantized_tensor(q8_tensor, q2_tensor):
                    matched_pairs.append((name, q8_tensor, q2_tensor))
                else:
                    unquantized_tensors.append(name)
            else:
                self.logger.warning(f"Tensor '{name}' found in Q8_0 but not in Q2_0")
        
        # Check for Q2_0 tensors not in Q8_0
        for name in q2_tensors:
            if name not in q8_tensors:
                self.logger.warning(f"Tensor '{name}' found in Q2_0 but not in Q8_0")
        
        self.logger.info(f"Found {len(matched_pairs)} quantized tensor pairs")
        self.logger.info(f"Found {len(unquantized_tensors)} unquantized tensors (kept as-is)")
        
        if self.verbose:
            print("\nMatched quantized tensors:")
            for name, q8_tensor, q2_tensor in matched_pairs[:10]:
                print(f"  {name}: {q8_tensor.shape} (Q8: {q8_tensor.size:,}B, Q2: {q2_tensor.size:,}B)")
            if len(matched_pairs) > 10:
                print(f"  ... and {len(matched_pairs) - 10} more")
        
        return matched_pairs
    
    def _is_quantized_tensor(self, q8_tensor: TensorInfo, q2_tensor: TensorInfo) -> bool:
        """
        Check if a tensor pair represents quantized weights.
        
        Validates that q8_tensor is Q8_0 and q2_tensor is a valid 2-bit quantization
        (Q2_K, IQ2_XXS, IQ2_XS, or Q2_0). Excludes embedding layers, output layers, 
        and other non-quantized tensors that should remain in their original precision.
        """
        name = q8_tensor.name.lower()
        
        # Tensors that are typically not quantized
        exclude_patterns = [
            'token_embd',     # Token embeddings
            'output_norm',    # Output normalization
            'output',         # Output projection (sometimes)
            'embed_tokens',   # Alternative embedding name
            'lm_head',        # Language model head
        ]
        
        for pattern in exclude_patterns:
            if pattern in name:
                return False
        
        # Check if both tensors are actually quantized
        from gguf_parser import GGMLType
        q8_quantized = q8_tensor.ggml_type == GGMLType.Q8_0
        
        # Accept various 2-bit quantization types
        valid_q2_types = {GGMLType.Q2_K, GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.Q2_0}
        q2_quantized = q2_tensor.ggml_type in valid_q2_types
        
        return q8_quantized and q2_quantized
    
    def build_mpgguf(self) -> None:
        """Build the MPGGUF file"""
        self.logger.info("Building MPGGUF file...")
        
        # Match tensors
        tensor_pairs = self.match_tensors()
        
        if not tensor_pairs:
            raise ValueError("No matching quantized tensors found between input files")
        
        # Create shared metadata
        self.logger.info("Creating shared metadata...")
        shared_metadata = self.format.create_shared_metadata(self.q8_parser, self.q2_parser)
        
        # Calculate tensor pair layout
        self.logger.info("Calculating interleaved tensor layout...")
        mpgguf_pairs = MPGGUFFormat.calculate_interleaved_offsets(tensor_pairs)
        
        # Calculate file layout
        header_size = MPGGUFHeader().size
        metadata_size = len(shared_metadata)
        metadata_aligned = MPGGUFFormat.align_size(metadata_size)
        
        # Calculate tensor pair index size
        index_size = self._calculate_tensor_index_size(mpgguf_pairs)
        index_aligned = MPGGUFFormat.align_size(index_size)
        
        # Tensor data starts after header + metadata + index
        tensor_data_offset = header_size + metadata_aligned + index_aligned
        tensor_data_offset = MPGGUFFormat.align_size(tensor_data_offset)
        
        # Create header
        header = MPGGUFHeader(
            tensor_pair_count=len(mpgguf_pairs),
            metadata_size=metadata_aligned,
            tensor_data_offset=tensor_data_offset
        )
        
        # Write MPGGUF file
        self._write_mpgguf_file(header, shared_metadata, mpgguf_pairs, tensor_pairs)
        
        # Calculate final size and compression ratio
        output_size = self.output_path.stat().st_size
        input_size = self.q8_path.stat().st_size + self.q2_path.stat().st_size
        
        self.logger.info(f"MPGGUF file created successfully!")
        self.logger.info(f"Output size: {output_size:,} bytes")
        self.logger.info(f"Input combined size: {input_size:,} bytes")
        self.logger.info(f"Size efficiency: {output_size/input_size:.2%} of combined inputs")
    
    def _calculate_tensor_index_size(self, pairs: List[MPGGUFTensorPair]) -> int:
        """Calculate the size needed for tensor pair index"""
        size = 0
        for pair in pairs:
            # Name: 8 bytes (length) + name bytes
            size += 8 + len(pair.name.encode('utf-8'))
            # Shape: 4 bytes (ndims) + 8 bytes per dimension  
            size += 4 + len(pair.shape) * 8
            # Offsets and sizes: 4 * 8 bytes
            size += 32
        return size
    
    def _write_mpgguf_file(self, header: MPGGUFHeader, metadata: bytes, 
                          mpgguf_pairs: List[MPGGUFTensorPair],
                          tensor_pairs: List[Tuple[str, TensorInfo, TensorInfo]]) -> None:
        """Write the complete MPGGUF file"""
        self.logger.info("Writing MPGGUF file...")
        
        with open(self.output_path, 'wb') as out_file:
            # Write header
            out_file.write(header.to_bytes())
            
            # Write shared metadata (with padding)
            out_file.write(metadata)
            metadata_padding = header.metadata_size - len(metadata)
            if metadata_padding > 0:
                out_file.write(b'\0' * metadata_padding)
            
            # Write tensor pair index
            self.format.write_tensor_pair_index(out_file, mpgguf_pairs)
            
            # Align to tensor data offset
            current_pos = out_file.tell()
            padding_needed = header.tensor_data_offset - current_pos
            if padding_needed > 0:
                out_file.write(b'\0' * padding_needed)
            
            # Write interleaved tensor data
            self._write_tensor_data(out_file, tensor_pairs, mpgguf_pairs)
    
    def _write_tensor_data(self, out_file: BinaryIO, 
                          tensor_pairs: List[Tuple[str, TensorInfo, TensorInfo]],
                          mpgguf_pairs: List[MPGGUFTensorPair]) -> None:
        """Write interleaved tensor data"""
        self.logger.info("Writing interleaved tensor data...")
        
        with open(self.q8_path, 'rb') as q8_file, open(self.q2_path, 'rb') as q2_file:
            for i, ((name, q8_tensor, q2_tensor), mpgguf_pair) in enumerate(zip(tensor_pairs, mpgguf_pairs)):
                if i % 100 == 0:
                    self.logger.debug(f"Writing tensor pair {i+1}/{len(tensor_pairs)}: {name}")
                
                # Read Q8_0 data
                q8_file.seek(self.q8_parser.tensor_data_offset + q8_tensor.offset)
                q8_data = q8_file.read(q8_tensor.size)
                
                # Read Q2_0 data
                q2_file.seek(self.q2_parser.tensor_data_offset + q2_tensor.offset)
                q2_data = q2_file.read(q2_tensor.size)
                
                # Write Q8_0 data first
                out_file.write(q8_data)
                
                # Write Q2_0 data immediately after
                out_file.write(q2_data)
                
                # Add padding to align next tensor pair
                current_pos = out_file.tell()
                next_aligned = MPGGUFFormat.align_size(current_pos)
                padding = next_aligned - current_pos
                if padding > 0:
                    out_file.write(b'\0' * padding)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Build Mixed-Precision GGUF (MPGGUF) files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python mpgguf_builder.py --q8 model.Q8_0.gguf --q2 model.Q2_0.gguf --out model.mpgguf
  
  # With verbose output
  python mpgguf_builder.py --q8 model.Q8_0.gguf --q2 model.Q2_0.gguf --out model.mpgguf -v
        """
    )
    
    parser.add_argument('--q8', required=True, help='Input Q8_0 GGUF file path')
    parser.add_argument('--q2', required=True, help='Input Q2_0 GGUF file path')  
    parser.add_argument('--out', required=True, help='Output MPGGUF file path')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        # Create builder and run
        builder = MPGGUFBuilder(args.q8, args.q2, args.out, args.verbose)
        builder.validate_inputs()
        builder.parse_input_files()
        builder.build_mpgguf()
        
        print(f"Success! MPGGUF file created: {args.out}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()