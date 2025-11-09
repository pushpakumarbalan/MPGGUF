/*
 * MPGGUF CUDA Dequantization Kernels
 * 
 * This file contains CUDA kernels for dequantizing Q8_0 and Q2_k tensors
 * back to FP16 format for validation against baseline models.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>

// Block sizes for different quantization formats
#define Q8_0_BLOCK_SIZE 32
#define Q2_0_BLOCK_SIZE 32

// Q8_0 block structure: 2 bytes (FP16 scale) + 32 bytes (int8 data)
struct block_q8_0 {
    half scale;        // FP16 scale factor
    int8_t qs[32];     // Quantized values (-128 to 127)
};

// Q2_0 block structure: 2 bytes (FP16 scale) + 8 bytes (2-bit packed data)
struct block_q2_0 {
    half scale;        // FP16 scale factor  
    uint8_t qs[8];     // 2-bit values packed (4 values per byte)
};

/**
 * CUDA kernel to dequantize Q8_0 format to FP16
 */
__global__ void dequantize_q8_0_kernel(
    const void* __restrict__ vx,    // Input Q8_0 data
    half* __restrict__ y,           // Output FP16 data
    const int k                     // Number of elements
) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i >= k) return;
    
    const int block_idx = i / Q8_0_BLOCK_SIZE;
    const int elem_idx = i % Q8_0_BLOCK_SIZE;
    // MPGGUF CUDA Dequantization Kernels
    //
    // Corrected implementation for Q8_0 and Q2_K dequantization.

    #include <cuda_runtime.h>
    #include <cuda_fp16.h>
    #include <stdio.h>
    #include <stdint.h>

    // Define ggml_half for the CUDA file
    typedef half ggml_half;
    typedef half2 ggml_half2;

    // K-quants
    #define QK_K 256
    #define Q8_0_BLOCK_SIZE 32

    // Q8_0 block structure
    struct block_q8_0 {
        half scale;
        int8_t qs[Q8_0_BLOCK_SIZE];
    };

    // 2-bit quantization (Q2_K)
    // 16 blocks of 16 elements each
    typedef struct {
        uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
        uint8_t qs[QK_K/4];      // quants
        union {
            struct {
                ggml_half d;      // super-block scale for quantized scales
                ggml_half dmin;   // super-block scale for quantized mins
            } ;
            ggml_half2 dm;
        } ;
    } block_q2_K;

    /**
     * CUDA kernel to dequantize Q8_0 format to FP16
     */
    __global__ void dequantize_q8_0_kernel(
        const void* __restrict__ vx,    // Input Q8_0 data
        half* __restrict__ y,           // Output FP16 data
        const int k                     // Number of elements
    ) {
        const int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= k) return;
    
        const int block_idx = i / Q8_0_BLOCK_SIZE;
        const int elem_idx = i % Q8_0_BLOCK_SIZE;
    
        const struct block_q8_0* x = (const struct block_q8_0*)vx;
        const struct block_q8_0* block = &x[block_idx];
    
        // Dequantize: value = scale * quantized_value
        y[i] = __hmul(block->scale, __int2half_rn(block->qs[elem_idx]));
    }

    /**
     * CUDA kernel to dequantize Q2_K format to FP16
     */
    __global__ void dequantize_q2_k_kernel(
        const void* __restrict__ vx,    // Input Q2_K data
        half* __restrict__ y,           // Output FP16 data
        const int k                     // Number of elements
    ) {
        const int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= k) return;

        const int super_block_idx = i / QK_K; // Index of the 256-element super-block
        const int elem_in_super_block = i % QK_K;

        const int sub_block_idx = elem_in_super_block / 16; // Index of the 16-element sub-block (0-15)
        const int elem_in_sub_block = elem_in_super_block % 16;

        const block_q2_K* x = (const block_q2_K*)vx;
        const block_q2_K* super_block = &x[super_block_idx];

        // 1. Get super-block scales (fp16)
        const float d = __half2float(super_block->d);
        const float dmin = __half2float(super_block->dmin);

        // 2. Get sub-block 4-bit scale and 4-bit min
        // scales[] stores 16 pairs of 4-bit (min, scale)
        const uint8_t scale_byte = super_block->scales[sub_block_idx];
        const uint8_t scale_nibble = scale_byte & 0x0F; // Low 4 bits
        const uint8_t min_nibble = (scale_byte >> 4) & 0x0F; // High 4 bits

        // 3. Dequantize sub-block scale and min
        const float sub_block_scale = d * scale_nibble;
        const float sub_block_min = dmin * min_nibble;

        // 4. Find the 2-bit quantized value
        const int qs_byte_idx = elem_in_super_block / 4; // Each byte holds 4 2-bit values
        const int qs_bit_offset = (elem_in_super_block % 4) * 2;
        const uint8_t qs_byte = super_block->qs[qs_byte_idx];
        const uint8_t q_val = (qs_byte >> qs_bit_offset) & 0x03; // Get the 2-bit value (0, 1, 2, or 3)

        // 5. Calculate final value: value = (sub_block_scale * q) + sub_block_min
        y[i] = __float2half((sub_block_scale * q_val) + sub_block_min);
    }

    // --- Host Functions ---
    extern "C" void dequantize_q8_0_cuda(
        const void* input,
        half* output,
        int num_elements
    ) {
        const int block_size = 256;
        const int grid_size = (num_elements + block_size - 1) / block_size;
        dequantize_q8_0_kernel<<<grid_size, block_size>>>(input, output, num_elements);
        cudaGetLastError(); // Check for errors
    }

    extern "C" void dequantize_q2_k_cuda(
        const void* input,
        half* output,
        int num_elements
    ) {
        const int block_size = 256;
        const int grid_size = (num_elements + block_size - 1) / block_size;
        dequantize_q2_k_kernel<<<grid_size, block_size>>>(input, output, num_elements);
        cudaGetLastError(); // Check for errors
    }

    // Kernel to convert half to float
    __global__ void half_to_float_kernel(const half* input, float* output, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) output[i] = __half2float(input[i]);
    }

    // Kernel to compute sum of squared errors
    __global__ void mse_kernel(const float* dequant, const float* baseline, float* sum_sq_err, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float diff = dequant[idx] - baseline[idx];
            atomicAdd(sum_sq_err, diff * diff);
        }
    }

    // Host function to compute MSE for Q8_0
    extern "C" void compute_mse_q8_0_cuda(const float* baseline, const uint8_t* q8_data, size_t num_elements, double* mse_out, double* rmse_out) {
        float* d_baseline;
        half* d_dequant_half;
        float* d_dequant;
        float* d_sum_sq_err;
        
        cudaMalloc(&d_baseline, num_elements * sizeof(float));
        cudaMalloc(&d_dequant_half, num_elements * sizeof(half));
        cudaMalloc(&d_dequant, num_elements * sizeof(float));
        cudaMalloc(&d_sum_sq_err, sizeof(float));
        
        cudaMemcpy(d_baseline, baseline, num_elements * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_sum_sq_err, 0, sizeof(float));
        
        // Dequantize
        dequantize_q8_0_cuda((const void*)q8_data, d_dequant_half, num_elements);
        
        // Convert to float
        int block_size = 256;
        int grid_size = (num_elements + block_size - 1) / block_size;
        half_to_float_kernel<<<grid_size, block_size>>>(d_dequant_half, d_dequant, num_elements);
        
        // Compute MSE
        mse_kernel<<<grid_size, block_size>>>(d_dequant, d_baseline, d_sum_sq_err, num_elements);
        
        float sum_sq_err;
        cudaMemcpy(&sum_sq_err, d_sum_sq_err, sizeof(float), cudaMemcpyDeviceToHost);
        
        double mse = sum_sq_err / num_elements;
        double rmse = sqrt(mse);
        *mse_out = mse;
        *rmse_out = rmse;
        
        cudaFree(d_baseline);
        cudaFree(d_dequant_half);
        cudaFree(d_dequant);
        cudaFree(d_sum_sq_err);
    }

    // Host function to compute MSE for Q2_K
    extern "C" void compute_mse_q2_k_cuda(const float* baseline, const uint8_t* q2_data, size_t num_elements, double* mse_out, double* rmse_out) {
        float* d_baseline;
        half* d_dequant_half;
        float* d_dequant;
        float* d_sum_sq_err;
        
        cudaMalloc(&d_baseline, num_elements * sizeof(float));
        cudaMalloc(&d_dequant_half, num_elements * sizeof(half));
        cudaMalloc(&d_dequant, num_elements * sizeof(float));
        cudaMalloc(&d_sum_sq_err, sizeof(float));
        
        cudaMemcpy(d_baseline, baseline, num_elements * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_sum_sq_err, 0, sizeof(float));
        
        // Dequantize
        dequantize_q2_k_cuda((const void*)q2_data, d_dequant_half, num_elements);
        
        // Convert to float
        int block_size = 256;
        int grid_size = (num_elements + block_size - 1) / block_size;
        half_to_float_kernel<<<grid_size, block_size>>>(d_dequant_half, d_dequant, num_elements);
        
        // Compute MSE
        mse_kernel<<<grid_size, block_size>>>(d_dequant, d_baseline, d_sum_sq_err, num_elements);
        
        float sum_sq_err;
        cudaMemcpy(&sum_sq_err, d_sum_sq_err, sizeof(float), cudaMemcpyDeviceToHost);
        
        double mse = sum_sq_err / num_elements;
        double rmse = sqrt(mse);
        *mse_out = mse;
        *rmse_out = rmse;
        
        cudaFree(d_baseline);
        cudaFree(d_dequant_half);
        cudaFree(d_dequant);
        cudaFree(d_sum_sq_err);
    }
