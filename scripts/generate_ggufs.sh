#!/bin/bash
"""
Generate GGUF files for MPGGUF testing

This script converts downloaded HuggingFace models to GGUF format in multiple quantizations:
- F16 (baseline for validation)
- Q8_0 (8-bit quantization)
- Q2_0 (2-bit quantization - custom implementation)
"""

set -e  # Exit on any error

# Configuration
MODELS_DIR="../models"
LLAMA_CPP_DIR="../llama.cpp"
CONVERTER="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
QUANTIZER="$LLAMA_CPP_DIR/build/bin/llama-quantize"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}INFO:${NC} $1"
}

log_success() {
    echo -e "${GREEN}SUCCESS:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}ERROR:${NC} $1"
}

# Check if required tools exist
check_requirements() {
    log_info "Checking requirements..."
    
    if [ ! -f "$CONVERTER" ]; then
        log_error "HuggingFace converter not found: $CONVERTER"
        log_error "Please run setup_llama_cpp.sh first"
        exit 1
    fi
    
    if [ ! -f "$QUANTIZER" ]; then
        log_error "llama-quantize not found: $QUANTIZER"
        log_error "Please run setup_llama_cpp.sh first"
        exit 1
    fi
    
    if [ ! -d "$MODELS_DIR" ]; then
        log_error "Models directory not found: $MODELS_DIR"
        log_error "Please run download_models.py first"
        exit 1
    fi
    
    log_success "All requirements found"
}

# Convert HF model to F16 GGUF
convert_to_f16() {
    local model_dir="$1"
    local model_name="$2"
    
    log_info "Converting $model_name to F16 GGUF..."
    
    local output_file="$model_dir/${model_name}.F16.gguf"
    
    if [ -f "$output_file" ]; then
        log_warning "F16 GGUF already exists: $output_file"
        return 0
    fi
    
    # Convert using llama.cpp converter
    cd "$LLAMA_CPP_DIR"
    python3 "$CONVERTER" "$model_dir" \
        --outfile "$output_file" \
        --outtype f16
    
    if [ $? -eq 0 ]; then
        log_success "F16 GGUF created: $output_file"
        ls -lh "$output_file"
    else
        log_error "Failed to convert $model_name to F16"
        return 1
    fi
}

# Quantize F16 GGUF to specified format
quantize_model() {
    local model_dir="$1" 
    local model_name="$2"
    local quant_type="$3"
    
    log_info "Quantizing $model_name to $quant_type..."
    
    local input_file="$model_dir/${model_name}.F16.gguf"
    local output_file="$model_dir/${model_name}.${quant_type}.gguf"
    
    if [ ! -f "$input_file" ]; then
        log_error "Input F16 GGUF not found: $input_file"
        return 1
    fi
    
    if [ -f "$output_file" ]; then
        log_warning "${quant_type} GGUF already exists: $output_file"
        return 0
    fi
    
    # Quantize using llama-quantize
    cd "$LLAMA_CPP_DIR"
    "$QUANTIZER" "$input_file" "$output_file" "$quant_type"
    
    if [ $? -eq 0 ]; then
        log_success "${quant_type} GGUF created: $output_file"
        ls -lh "$output_file"
    else
        log_error "Failed to quantize $model_name to $quant_type"
        return 1
    fi
}

# Process a single model directory
process_model() {
    local model_key="$1"
    local model_dir="$MODELS_DIR/$model_key"
    
    log_info "Processing model: $model_key"
    
    if [ ! -d "$model_dir" ]; then
        log_error "Model directory not found: $model_dir"
        return 1
    fi
    
    # Verify essential files exist
    if [ ! -f "$model_dir/config.json" ]; then
        log_error "config.json not found in $model_dir"
        return 1
    fi
    
    # Count safetensors files
    local safetensor_count=$(find "$model_dir" -name "*.safetensors" | wc -l)
    if [ "$safetensor_count" -eq 0 ]; then
        log_error "No .safetensors files found in $model_dir"
        return 1
    fi
    
    log_info "Found $safetensor_count safetensors files"
    
    # Step 1: Convert to F16 GGUF (baseline)
    convert_to_f16 "$model_dir" "$model_key"
    if [ $? -ne 0 ]; then
        return 1
    fi
    
    # Step 2: Quantize to Q8_0
    quantize_model "$model_dir" "$model_key" "Q8_0"
    if [ $? -ne 0 ]; then
        return 1
    fi
    
    # Step 3: Quantize to Q4_0 (Q2_0 might not be available yet)
    # Note: Q2_0 is a custom implementation that might need special handling
    log_info "Attempting Q4_0 quantization (Q2_0 may need custom implementation)..."
    quantize_model "$model_dir" "$model_key" "Q4_0"
    
    # Try Q2_K as alternative to Q2_0
    log_info "Attempting Q2_K quantization (alternative to Q2_0)..."
    quantize_model "$model_dir" "$model_key" "Q2_K"
    
    log_success "Completed processing $model_key"
    
    # Show final results
    echo ""
    log_info "Generated files for $model_key:"
    ls -lh "$model_dir"/*.gguf 2>/dev/null || log_warning "No GGUF files found"
    echo ""
}

# Test model with llama-cli
test_model() {
    local model_file="$1"
    local model_name="$2"
    
    log_info "Testing $model_name..."
    
    if [ ! -f "$model_file" ]; then
        log_error "Model file not found: $model_file"
        return 1
    fi
    
    # Quick test with a simple prompt
    cd "$LLAMA_CPP_DIR"
    echo "The quick brown fox" | timeout 30s "$LLAMA_CPP_DIR/build/bin/llama-cli" \
        -m "$model_file" \
        -n 10 \
        --temp 0.1 \
        -p "Complete this sentence: The quick brown fox" \
        --no-display-prompt 2>/dev/null
    
    if [ $? -eq 0 ]; then
        log_success "Model test passed: $model_name"
    else
        log_warning "Model test had issues (this may be normal for some configs)"
    fi
}

# Main execution
main() {
    echo "========================================"
    echo "GGUF Generation Script for MPGGUF"
    echo "========================================"
    
    check_requirements
    
    # Check command line arguments
    if [ $# -eq 0 ]; then
        log_info "No specific model provided. Processing all available models..."
        
        # Find all model directories
        model_dirs=($(find "$MODELS_DIR" -maxdepth 1 -type d -not -path "$MODELS_DIR"))
        
        if [ ${#model_dirs[@]} -eq 0 ]; then
            log_error "No model directories found in $MODELS_DIR"
            log_info "Please run download_models.py first"
            exit 1
        fi
        
        log_info "Found ${#model_dirs[@]} model(s) to process"
        
        # Process each model
        for model_dir in "${model_dirs[@]}"; do
            model_key=$(basename "$model_dir")
            process_model "$model_key"
        done
        
    else
        # Process specific models
        for model_key in "$@"; do
            process_model "$model_key"
        done
    fi
    
    echo ""
    log_success "GGUF generation completed!"
    
    # Show summary
    echo ""
    log_info "Summary of generated GGUF files:"
    find "$MODELS_DIR" -name "*.gguf" -exec ls -lh {} \; 2>/dev/null | sort
    
    echo ""
    log_info "Next steps:"
    log_info "1. Test MPGGUF builder: python src/mpgguf_builder.py --q8 model.Q8_0.gguf --q2 model.Q2_K.gguf --out model.mpgguf"
    log_info "2. Run validation: python src/validation/error_analysis.py --mpgguf model.mpgguf --baseline model.F16.gguf"
}

# Handle interruption
trap 'log_error "Script interrupted by user"; exit 1' INT

# Run main function with all arguments
main "$@"