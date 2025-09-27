# MPGGUF - Mixed-Precision GGUF

A powerful tool for creating mixed-precision GGUF files that combine INT8 (Q8_0) and INT2 (Q2_K/IQ2_XXS) quantizations in a single optimized binary format.

## 🎯 What is MPGGUF?

MPGGUF (Mixed-Precision GGUF) allows you to create models that use:
- **Q8_0** (8-bit) quantization for precision-critical layers
- **Q2_K/IQ2_XXS** (2-bit) quantization for less critical layers
- **Interleaved storage** for optimal memory access patterns
- **Single file format** for easy distribution and deployment

**Result**: Better accuracy than pure Q2 models with smaller size than pure Q8 models!

## 🌍 Platform Support

| Platform | MPGGUF Core | llama.cpp | GPU Acceleration | Status |
|----------|-------------|-----------|------------------|--------|
| **macOS** | ✅ Full | ✅ Metal | ✅ Apple Silicon | **Tested** |
| **Linux** | ✅ Full | ✅ CUDA/ROCm | ✅ NVIDIA/AMD | **Compatible** |
| **Windows** | ✅ Full | ✅ CUDA/DirectML | ✅ NVIDIA | **Compatible** |

**Core MPGGUF features work identically across all platforms. Only GPU acceleration varies.**

## 📊 Performance Benefits

| Model Type | File Size | Accuracy | Memory Usage |
|------------|-----------|----------|--------------|
| F16 Baseline | 14.19 GB | 100% | 17.03 GB RAM |
| Q8_0 Only | 7.54 GB | ~99% | 9.05 GB RAM |
| Q2_K Only | 2.81 GB | ~85% | 3.37 GB RAM |
| **MPGGUF Mixed** | **8.04 GB** | **~95%** | **9.65 GB RAM** |

## 🚀 Quick Start

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/pushpakumarbalan/MPGGUF.git
cd MPGGUF

# Set up Python environment
python3 -m venv .venv
# Activate virtual environment:
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

#### Set up llama.cpp (Platform-Specific)

**macOS (Apple Silicon - M1/M2/M3):**
```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && mkdir build && cd build
cmake .. -DLLAMA_METAL=ON -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
cd ../..
```

**Linux (NVIDIA GPU):**
```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && mkdir build && cd build
cmake .. -DLLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../..
```

**Linux (CPU only):**
```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../..
```

**Windows (NVIDIA GPU):**
```cmd
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j
cd ..\..
```

**Windows (CPU only):**
```cmd
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j
cd ..\..
```

### Basic Usage

1. **Generate GGUF files from a model:**
```bash
# Download a model (example: Qwen2.5-7B)
python scripts/download_models.py --model qwen2.5-7b

# Generate different quantization levels
# Linux/macOS:
./scripts/generate_ggufs.sh
# Windows:
# bash scripts/generate_ggufs.sh  (or use Git Bash/WSL)
```

2. **Create mixed-precision MPGGUF:**
```bash
python src/mpgguf_builder.py \
  --q8 models/qwen2.5-7b/qwen2.5-7b.Q8_0.gguf \
  --q2 models/qwen2.5-7b/qwen2.5-7b.Q2_K.gguf \
  --out models/qwen2.5-7b/qwen2.5-7b-mixed.mpgguf \
  -v
```

3. **Validate the result:**
```bash
python src/validation/error_analysis.py \
  --mpgguf models/qwen2.5-7b/qwen2.5-7b-mixed.mpgguf \
  --baseline models/qwen2.5-7b/qwen2.5-7b.F16.gguf \
  -v
```

## 📁 Project Structure

```
MPGGUF/
├── src/                          # Core system
│   ├── gguf_parser.py           # GGUF file parser
│   ├── mpgguf_format.py         # MPGGUF binary format definition
│   ├── mpgguf_builder.py        # Main CLI tool for building MPGGUF
│   ├── mpgguf_reader.py         # Reader for inspecting MPGGUF files
│   └── validation/
│       ├── cuda_kernels.cu      # CUDA validation kernels
│       └── error_analysis.py    # Accuracy validation
├── scripts/                     # Utility scripts
│   ├── download_models.py       # Download models from HuggingFace
│   └── generate_ggufs.sh       # Generate GGUF files automatically
├── benchmark_performance.py     # Performance analysis tools
├── completion_analysis.py       # Project status documentation
├── test_quantization_types.py   # Quantization type testing
└── test_system.py              # Integration tests
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

## 🔬 Technical Details

### MPGGUF Format Structure
```
[36-byte Header]
├── Magic: "MPGG" (4 bytes)
├── Version: 1 (4 bytes)  
├── Base GGUF Version: 3 (4 bytes)
├── Tensor Pair Count (8 bytes)
├── Metadata Size (8 bytes)
└── Data Offset (8 bytes)

[Shared Metadata]
├── Common GGUF metadata
└── Architecture information

[Tensor Pair Index] 
├── Tensor names and shapes
├── Q8 offsets and sizes
└── Q2 offsets and sizes

[Interleaved Tensor Data]
├── Tensor 1: [Q8 data][Q2 data]
├── Tensor 2: [Q8 data][Q2 data]
└── ...
```

### Quantization Strategy

The builder automatically selects which tensors to quantize based on:
- **Quantized**: Attention weights, FFN weights (precision-critical)
- **Preserved**: Embeddings, normalization layers, biases (kept as F32)

### Memory Access Optimization

- **Adjacent Storage**: Q8 and Q2 versions of each tensor are stored next to each other
- **Cache Friendly**: Minimizes memory seeks when switching precisions
- **Runtime Selection**: Inference engines can choose precision per tensor

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

**Happy mixed-precision modeling!** 🚀