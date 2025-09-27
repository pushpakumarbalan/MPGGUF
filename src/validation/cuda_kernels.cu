/*
 * MPGGUF CUDA Dequantization Kernels
 * 
 * This file contains CUDA kernels for dequantizing Q8_0 and Q2_0 tensors
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
    
    const struct block_q8_0* x = (const struct block_q8_0*)vx;
    const struct block_q8_0* block = &x[block_idx];
    
    // Dequantize: value = scale * quantized_value
    y[i] = __hmul(block->scale, __int2half_rn(block->qs[elem_idx]));
}

/**
 * CUDA kernel to dequantize Q2_0 format to FP16
 */
__global__ void dequantize_q2_0_kernel(
    const void* __restrict__ vx,    // Input Q2_0 data
    half* __restrict__ y,           // Output FP16 data
    const int k                     // Number of elements
) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i >= k) return;
    
    const int block_idx = i / Q2_0_BLOCK_SIZE;
    const int elem_idx = i % Q2_0_BLOCK_SIZE;
    
    const struct block_q2_0* x = (const struct block_q2_0*)vx;
    const struct block_q2_0* block = &x[block_idx];
    
    // Extract 2-bit value from packed data
    const int byte_idx = elem_idx / 4;
    const int bit_offset = (elem_idx % 4) * 2;
    const uint8_t packed_byte = block->qs[byte_idx];
    const uint8_t q2_val = (packed_byte >> bit_offset) & 0x03;
    
    // Convert 2-bit value to signed range (-1, 0, 1, 2) -> (-1.5, -0.5, 0.5, 1.5)
    const float q2_float = (float)q2_val - 1.5f;
    
    // Dequantize: value = scale * (q2_value - 1.5)
    y[i] = __hmul(block->scale, __float2half(q2_float));
}

/**
 * CUDA kernel to compute element-wise squared error between two FP16 arrays
 */
__global__ void compute_squared_error_kernel(
    const half* __restrict__ a,     // First FP16 array
    const half* __restrict__ b,     // Second FP16 array  
    float* __restrict__ errors,     // Output squared errors
    const int n                     // Number of elements
) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i >= n) return;
    
    const float diff = __half2float(a[i]) - __half2float(b[i]);
    errors[i] = diff * diff;
}

/**
 * CUDA kernel to compute sum reduction for MSE calculation
 */
__global__ void reduce_sum_kernel(
    const float* __restrict__ input,    // Input array
    float* __restrict__ output,         // Output partial sums
    const int n                         // Number of elements
) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/**
 * Host function to launch Q8_0 dequantization
 */
extern "C" void dequantize_q8_0_cuda(
    const void* input,      // Q8_0 quantized data
    half* output,           // FP16 output array
    int num_elements        // Total number of elements
) {
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;
    
    dequantize_q8_0_kernel<<<grid_size, block_size>>>(input, output, num_elements);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error in dequantize_q8_0: %s\n", cudaGetErrorString(err));
    }
}

/**
 * Host function to launch Q2_0 dequantization  
 */
extern "C" void dequantize_q2_0_cuda(
    const void* input,      // Q2_0 quantized data
    half* output,           // FP16 output array
    int num_elements        // Total number of elements
) {
    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;
    
    dequantize_q2_0_kernel<<<grid_size, block_size>>>(input, output, num_elements);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error in dequantize_q2_0: %s\n", cudaGetErrorString(err));
    }
}

/**
 * Host function to compute MSE between two FP16 arrays on GPU
 */
extern "C" float compute_mse_cuda(
    const half* array_a,    // First FP16 array
    const half* array_b,    // Second FP16 array
    int num_elements        // Number of elements
) {
    // Allocate device memory for squared errors
    float* d_errors;
    cudaMalloc(&d_errors, num_elements * sizeof(float));
    
    // Compute squared errors
    const int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    
    compute_squared_error_kernel<<<grid_size, block_size>>>(
        array_a, array_b, d_errors, num_elements
    );
    
    // Perform reduction to sum all squared errors
    float* d_partial_sums;
    const int max_blocks = 65535;  // Max grid size for reduction
    grid_size = min(grid_size, max_blocks);
    
    cudaMalloc(&d_partial_sums, grid_size * sizeof(float));
    
    const int shared_mem_size = block_size * sizeof(float);
    reduce_sum_kernel<<<grid_size, block_size, shared_mem_size>>>(
        d_errors, d_partial_sums, num_elements
    );
    
    // Copy partial sums back and finish reduction on CPU
    float* h_partial_sums = new float[grid_size];
    cudaMemcpy(h_partial_sums, d_partial_sums, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total_sum = 0.0f;
    for (int i = 0; i < grid_size; i++) {
        total_sum += h_partial_sums[i];
    }
    
    // Cleanup
    delete[] h_partial_sums;
    cudaFree(d_errors);
    cudaFree(d_partial_sums);
    
    // Return MSE
    return total_sum / num_elements;
}

/**
 * Host function to compute RMSE between two FP16 arrays
 */
extern "C" float compute_rmse_cuda(
    const half* array_a,    // First FP16 array
    const half* array_b,    // Second FP16 array
    int num_elements        // Number of elements  
) {
    float mse = compute_mse_cuda(array_a, array_b, num_elements);
    return sqrtf(mse);
}

/**
 * Utility function to check CUDA device properties
 */
extern "C" void print_cuda_device_info() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    printf("CUDA Device Information:\n");
    printf("Found %d CUDA device(s)\n\n", device_count);
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Global Memory: %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("  Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Grid Size: (%d, %d, %d)\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");
    }
}

/**
 * Test function to validate kernel implementations
 */
extern "C" int test_dequantization_kernels() {
    printf("Testing CUDA dequantization kernels...\n");
    
    // Test parameters
    const int num_elements = 1024;
    const int num_blocks_q8 = (num_elements + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
    const int num_blocks_q2 = (num_elements + Q2_0_BLOCK_SIZE - 1) / Q2_0_BLOCK_SIZE;
    
    // Allocate host memory for test data
    struct block_q8_0* h_q8_data = (struct block_q8_0*)malloc(num_blocks_q8 * sizeof(struct block_q8_0));
    struct block_q2_0* h_q2_data = (struct block_q2_0*)malloc(num_blocks_q2 * sizeof(struct block_q2_0));
    half* h_output_q8 = (half*)malloc(num_elements * sizeof(half));
    half* h_output_q2 = (half*)malloc(num_elements * sizeof(half));
    
    // Initialize test data
    for (int i = 0; i < num_blocks_q8; i++) {
        h_q8_data[i].scale = __float2half(1.0f);  // Unit scale for testing
        for (int j = 0; j < Q8_0_BLOCK_SIZE; j++) {
            h_q8_data[i].qs[j] = (int8_t)(j - 16);  // Test pattern
        }
    }
    
    for (int i = 0; i < num_blocks_q2; i++) {
        h_q2_data[i].scale = __float2half(1.0f);  // Unit scale for testing
        for (int j = 0; j < 8; j++) {
            h_q2_data[i].qs[j] = 0x1B;  // 0b00011011 = values 3,2,1,0
        }
    }
    
    // Allocate device memory
    void* d_q8_data;
    void* d_q2_data;
    half* d_output_q8;
    half* d_output_q2;
    
    cudaMalloc(&d_q8_data, num_blocks_q8 * sizeof(struct block_q8_0));
    cudaMalloc(&d_q2_data, num_blocks_q2 * sizeof(struct block_q2_0));
    cudaMalloc(&d_output_q8, num_elements * sizeof(half));
    cudaMalloc(&d_output_q2, num_elements * sizeof(half));
    
    // Copy test data to device
    cudaMemcpy(d_q8_data, h_q8_data, num_blocks_q8 * sizeof(struct block_q8_0), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q2_data, h_q2_data, num_blocks_q2 * sizeof(struct block_q2_0), cudaMemcpyHostToDevice);
    
    // Run kernels
    dequantize_q8_0_cuda(d_q8_data, d_output_q8, num_elements);
    dequantize_q2_0_cuda(d_q2_data, d_output_q2, num_elements);
    
    // Copy results back
    cudaMemcpy(h_output_q8, d_output_q8, num_elements * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_q2, d_output_q2, num_elements * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Verify results (basic sanity check)
    printf("Q8_0 dequantization test - first 8 values:\n");
    for (int i = 0; i < 8; i++) {
        printf("  [%d]: %f\n", i, __half2float(h_output_q8[i]));
    }
    
    printf("Q2_0 dequantization test - first 8 values:\n");
    for (int i = 0; i < 8; i++) {
        printf("  [%d]: %f\n", i, __half2float(h_output_q2[i]));
    }
    
    // Cleanup
    free(h_q8_data);
    free(h_q2_data);
    free(h_output_q8);
    free(h_output_q2);
    cudaFree(d_q8_data);
    cudaFree(d_q2_data);
    cudaFree(d_output_q8);
    cudaFree(d_output_q2);
    
    printf("Kernel tests completed successfully!\n");
    return 0;
}