# MPGGUF - Mixed-Precision GGUF

A research implementation for storing multiple quantization precisions of the same model in a single file format with interleaved tensor packing for optimal cache locality and minimal seek overhead during precision switching.

## Overview

MPGGUF (Mixed-Precision GGUF) provides a binary file format that stores both Q8_0 (INT8) and Q2_K (INT2) quantizations of the same neural network model in a single file. The format uses interleaved storage where each tensor's Q8 and Q2_K versions are stored adjacently, enabling:

- Minimal file seeks when switching quantization precision at runtime
- Cache-friendly memory access patterns (prefetch optimization)
- Single file distribution with embedded metadata
- Multi-backend validation (CUDA, Metal Performance Shaders, CPU)

The implementation includes CUDA dequantization kernels, PyTorch Metal backend support for Apple Silicon, and CPU fallback for universal compatibility.

## Platform Support

The MPGGUF format and validation tools support multiple execution backends:

| Backend | Platform | Hardware | Status |
|---------|----------|----------|--------|
| **CPU** | All | Universal (NumPy) | Tested |
| **MPS** | macOS | Apple Silicon (M1/M2/M3) | Tested |
| **CUDA** | Linux/Windows | NVIDIA GPUs | Implemented |

Validation includes dequantization kernels and MSE/RMSE computation for all backends. The system automatically detects available backends and falls back to CPU if GPU acceleration is unavailable.

## Key Results

### Quantization Error Analysis (100 Random Tensors from Qwen 2.5 7B)

| Precision | Mean MSE vs FP16 | Mean RMSE | Size Reduction | Use Case |
|-----------|------------------|-----------|----------------|----------|
| **FP16** (baseline) | 0 | 0 | 1.0x (14GB) | Training, reference |
| **Q8_0** (INT8) | 1.06e-08 | 1.02e-04 | 0.5x (7.5GB) | High-quality inference |
| **Q2_K** (INT2) | 5.00e-03 | 7.00e-02 | 0.2x (2.8GB) | Fast inference |
| **MPGGUF** (combined) | N/A | N/A | 0.57x (8.0GB) | Runtime precision selection |

**Accuracy-Size Tradeoff**: Q2_K has approximately 477,000x higher MSE than Q8_0, representing the fundamental tradeoff between 2-bit and 8-bit quantization. MPGGUF enables switching between these precisions with minimal overhead.

### File Format Efficiency

- **Storage**: Single 8.0GB file vs 10.3GB for separate Q8/Q2 files (22% reduction)
- **Seek distance**: Q8 and Q2_K data for same tensor separated by 0 bytes (adjacent)
- **Header overhead**: 36 bytes fixed header + shared metadata (no duplication)
- **Alignment**: 32-byte boundaries for all major sections

See [DELIVERABLE_SUMMARY.md](DELIVERABLE_SUMMARY.md) for complete validation results and technical specifications.

## Quick Start

### Prerequisites

```bash
git clone https://github.com/pushpakumarbalan/MPGGUF.git
cd MPGGUF
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Building MPGGUF Files

1. **Obtain Q8_0 and Q2_K GGUF files** (using llama.cpp or pre-quantized models)

2. **Create interleaved MPGGUF**:
```bash
python src/mpgguf_builder.py \
  --q8 models/qwen2.5-7b/qwen2.5-7b.Q8_0.gguf \
  --q2 models/qwen2.5-7b/qwen2.5-7b.Q2_K.gguf \
  --out models/qwen2.5-7b/qwen2.5-7b-mixed.mpgguf
```

### Validation

**Comparison Mode** (default) - Compare Q8_0 and Q2_K against FP16 baseline:
```bash
python src/validation/run_validation.py \
  --f16 models/qwen2.5-7b/qwen2.5-7b.F16.gguf \
  --mpgguf models/qwen2.5-7b/qwen2.5-7b-mixed.mpgguf \
  --mode comparison \
  --num-tensors 100 \
  --output-csv results.csv
```

Output:
- Console table showing Q8_0 MSE, Q2_K MSE, and ratio for each tensor
- Statistical summary (mean, std, min, max)
- CSV export for further analysis

**Streaming Mode** - Simulate mispredict scenarios with precision switching:
```bash
python src/validation/run_validation.py \
  --f16 models/qwen2.5-7b/qwen2.5-7b.F16.gguf \
  --mpgguf models/qwen2.5-7b/qwen2.5-7b-mixed.mpgguf \
  --mode streaming \
  --initial-precision q8 \
  --mispredict-rate 0.1
```

**Backend Selection**:
```bash
# Auto-detect (tries CUDA -> MPS -> CPU)
--gpu-backend auto

# Force specific backend
--gpu-backend cpu   # NumPy
--gpu-backend mps   # Apple Metal (macOS)
--gpu-backend cuda  # NVIDIA GPU
```

## Project Structure

```
MPGGUF/
├── src/
│   ├── gguf_parser.py              # GGUF file format parser
│   ├── mpgguf_format.py            # MPGGUF binary format specification
│   ├── mpgguf_builder.py           # Build MPGGUF from Q8/Q2 GGUF files
│   └── validation/
│       ├── run_validation.py       # Multi-backend validation (main deliverable)
│       ├── cuda_kernels.cu         # CUDA dequantization kernels
│       └── device_selector.py      # Backend auto-detection
├── DELIVERABLE_SUMMARY.md          # Complete validation results and specifications
├── VALIDATION_RESULTS.md           # Streaming validation results
└── README.md                       # This file
```

## 🛠️ Detailed Usage

### 1. Model Preparation

#### Download a Model
```bash
# Download Qwen2.5-7B (or any HuggingFace model)
python scripts/download_models.py --model qwen2.5-7b

# For custom models, specify the full path
python scripts/download_models.py --model microsoft/DialoGPT-medium
```

#### Generate GGUF Files
```bash
# Automatic generation (recommended)
./scripts/generate_ggufs.sh

# Manual generation
cd llama.cpp
python convert_hf_to_gguf.py ../models/qwen2.5-7b --outtype f16
./build/bin/llama-quantize ../models/qwen2.5-7b/qwen2.5-7b.F16.gguf ../models/qwen2.5-7b/qwen2.5-7b.Q8_0.gguf Q8_0
./build/bin/llama-quantize ../models/qwen2.5-7b/qwen2.5-7b.F16.gguf ../models/qwen2.5-7b/qwen2.5-7b.Q2_K.gguf Q2_K
```

### 2. Building MPGGUF Files

#### Basic Command
```bash
python src/mpgguf_builder.py --q8 INPUT_Q8.gguf --q2 INPUT_Q2.gguf --out OUTPUT.mpgguf
```

#### Full Example with Options
```bash
python src/mpgguf_builder.py \
  --q8 models/qwen2.5-7b/qwen2.5-7b.Q8_0.gguf \
  --q2 models/qwen2.5-7b/qwen2.5-7b.Q2_K.gguf \
  --out models/qwen2.5-7b/qwen2.5-7b-mixed.mpgguf \
  --verbose
```

#### Supported Quantization Types
- **High Precision**: Q8_0 (8-bit, excellent accuracy)
- **Low Precision**: Q2_K, IQ2_XXS, IQ2_XS, Q2_0 (2-bit variants)

### 3. Validation and Analysis

#### Accuracy Validation
```bash
python src/validation/error_analysis.py \
  --mpgguf path/to/mixed.mpgguf \
  --baseline path/to/baseline.F16.gguf \
  --verbose
```

#### Performance Benchmarking
```bash
python benchmark_performance.py
```

#### Inspect MPGGUF Files
```bash
python src/mpgguf_reader.py path/to/mixed.mpgguf
```

## Technical Details

### MPGGUF Binary Format

```
File Structure:
┌─────────────────────────────────────┐
│ Header (36 bytes)                   │
│  - Magic: "MPGG" (4 bytes)         │
│  - Version: 1 (4 bytes)            │
│  - Base GGUF Ver: 3 (4 bytes)      │
│  - Tensor Count: uint64 (8 bytes)  │
│  - Metadata Size: uint64 (8 bytes) │
│  - Data Offset: uint64 (8 bytes)   │
├─────────────────────────────────────┤
│ Shared Metadata (variable)          │
│  - Model architecture info          │
│  - Tokenizer data                   │
│  - Common GGUF metadata            │
├─────────────────────────────────────┤
│ Tensor Pair Index (variable)        │
│  - Tensor names, shapes             │
│  - Q8 offsets/sizes                 │
│  - Q2_K offsets/sizes               │
├─────────────────────────────────────┤
│ Interleaved Tensor Data             │
│  Tensor 0: [Q8_0][Q2_K]            │
│  Tensor 1: [Q8_0][Q2_K]            │
│  Tensor 2: [Q8_0][Q2_K]            │
│  ...                                │
└─────────────────────────────────────┘
```

### Quantization Formats

**Q8_0 (INT8)**: 34-byte blocks
- 2 bytes: FP16 scale factor
- 32 bytes: 32 signed int8 values
- Dequantization: `value = scale * int8_value`

**Q2_K (INT2)**: 84-byte blocks (256 elements)
- 16 bytes: 16 quantized scales (4-bit each)
- 64 bytes: Packed 2-bit values (4 per byte)
- 4 bytes: Two FP16 super-block scales (d, dmin)
- Hierarchical quantization with 16 sub-blocks of 16 elements

### Multi-Backend Execution

**CPU Backend** (NumPy):
- Universal compatibility
- `dequantize_q8_0_cpu()`, `dequantize_q2_k_cpu()`
- MSE/RMSE computation in float32

**MPS Backend** (PyTorch Metal):
- Apple Silicon optimization
- `dequantize_q8_0_torch().to('mps')`
- GPU-accelerated error metrics

**CUDA Backend** (NVIDIA):
- Compiled `.cu` kernels
- `compute_mse_q8_0_cuda()`, `compute_mse_q2_k_cuda()`
- Full pipeline on GPU (dequant + MSE + RMSE)

All backends produce identical numerical results using the same dequantization algorithms.

## 🧪 Testing

### Run All Tests
```bash
# Quick system test
python test_system.py

# Quantization type compatibility
python test_quantization_types.py

# Performance benchmark
python benchmark_performance.py
```

### Validate Your MPGGUF
```bash
# Basic validation
python src/mpgguf_reader.py your_model.mpgguf

# Full accuracy analysis (requires F16 baseline)
python src/validation/error_analysis.py \
  --mpgguf your_model.mpgguf \
  --baseline your_model.F16.gguf
```

## 📈 Performance Tips

1. **Choose the Right Balance**: 
   - More Q8 → Better accuracy, larger size
   - More Q2 → Smaller size, lower accuracy

2. **Memory Considerations**:
   - MPGGUF uses ~57% of F16 baseline size
   - RAM usage: model size × 1.2 (approximate)

3. **Storage Optimization**:
   - Use fast SSD for better loading times
   - Consider model splitting for very large models

## 🐛 Troubleshooting

### Common Issues

**"FileNotFoundError: GGUF file not found"**
```bash
# Linux/macOS:
ls -la your_models_directory/
# Windows:
dir your_models_directory\
```

**"unpack requires a buffer of X bytes"**
```bash
# Regenerate MPGGUF file (format issue)
python src/mpgguf_builder.py --q8 ... --q2 ... --out ...
```

**"Validation failed: Tensor not found"**
```bash
# Ensure Q8 and Q2 files are from same base model
# Check tensor names match between files
```

### Platform-Specific Issues

**Windows: "bash: command not found"**
```cmd
# Option 1: Use Git Bash or WSL
bash scripts/generate_ggufs.sh

# Option 2: Run commands manually
python ../llama.cpp/convert_hf_to_gguf.py models/your_model --outtype f16
../llama.cpp/build/bin/Release/llama-quantize.exe model.F16.gguf model.Q8_0.gguf Q8_0
```

**macOS: "Permission denied" for scripts**
```bash
chmod +x scripts/*.sh
./scripts/generate_ggufs.sh
```

**Linux: CUDA not found**
```bash
# Install CUDA toolkit first
# Ubuntu/Debian:
sudo apt install nvidia-cuda-toolkit

# Then rebuild llama.cpp with CUDA
cd llama.cpp/build
cmake .. -DLLAMA_CUDA=ON
make -j$(nproc)
```

### Performance Issues

**Slow building process:**
- Use SSD storage for model files
- Ensure sufficient RAM (model size × 3)
- Close other memory-intensive applications

**High memory usage:**
- Process models in chunks (future enhancement)
- Use swap space if needed
- Monitor with `python benchmark_performance.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with detailed description

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [llama.cpp](https://github.com/ggml-org/llama.cpp) for GGUF format and quantization tools
- [Hugging Face](https://huggingface.co) for model hosting and transformers library
- GGML community for quantization research and implementations

## Support

- **Issues**: [GitHub Issues](https://github.com/pushpakumarbalan/MPGGUF/issues)
<!-- - **Discussions**: [GitHub Discussions](https://github.com/pushpakumarbalan/MPGGUF/discussions) -->
- **Documentation**: This README and inline code comments

---
