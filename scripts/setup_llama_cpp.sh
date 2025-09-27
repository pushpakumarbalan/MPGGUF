#!/bin/bash
"""
Setup script for llama.cpp and dependencies

This script downloads, builds, and sets up llama.cpp for GGUF generation.
"""

set -e  # Exit on any error

# Configuration
LLAMA_CPP_DIR="../llama.cpp"
MODELS_DIR="../models"
BUILD_DIR="$LLAMA_CPP_DIR/build"

echo "========================================="
echo "Setting up llama.cpp for MPGGUF project"
echo "========================================="

# Create directories
mkdir -p "$MODELS_DIR"

# Check if llama.cpp already exists
if [ -d "$LLAMA_CPP_DIR" ]; then
    echo "llama.cpp directory already exists. Pulling latest changes..."
    cd "$LLAMA_CPP_DIR"
    git pull origin master
    cd -
else
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
fi

cd "$LLAMA_CPP_DIR"

echo "Installing Python dependencies..."
python3 -m pip install -r requirements.txt

# Detect platform and set build options
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS - building with Metal support"
    BUILD_OPTS="-DLLAMA_METAL=ON"
elif command -v nvidia-smi &> /dev/null; then
    echo "Detected NVIDIA GPU - building with CUDA support"
    BUILD_OPTS="-DLLAMA_CUDA=ON"
else
    echo "Building CPU-only version"
    BUILD_OPTS=""
fi

# Build llama.cpp
echo "Building llama.cpp..."
mkdir -p build
cd build

cmake .. $BUILD_OPTS \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON

make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "Build completed!"

# Verify executables
cd ..
if [ -f "build/bin/llama-cli" ]; then
    echo "✓ llama-cli built successfully"
else
    echo "✗ llama-cli not found"
    exit 1
fi

if [ -f "build/bin/llama-quantize" ]; then
    echo "✓ llama-quantize built successfully"
else
    echo "✗ llama-quantize not found"
    exit 1
fi

# Test llama.cpp version
echo "Testing llama.cpp installation..."
./build/bin/llama-cli --version

echo ""
echo "========================================="
echo "llama.cpp setup completed successfully!"
echo "========================================="
echo ""
echo "Executables available:"
echo "  Convert: $PWD/convert_hf_to_gguf.py"
echo "  Quantize: $PWD/build/bin/llama-quantize"
echo "  Test: $PWD/build/bin/llama-cli"
echo ""
echo "Models directory: $(realpath "$MODELS_DIR")"
echo ""
echo "Next steps:"
echo "  1. Run ./scripts/download_models.py to download target models"
echo "  2. Run ./scripts/generate_ggufs.sh to create quantized versions"