#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256  // Number of threads per block
#define RADIUS 1        // Stencil radius

// CUDA kernel for a simple 1D stencil computation
__global__ void stencil_1d(float *d_out, float *d_in, int n) {
    __shared__ float smem[BLOCK_SIZE + 2 * RADIUS]; // Shared memory with a halo region

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread index
    int localIdx  = threadIdx.x + RADIUS;                  // Local index in shared memory

    // Load elements into shared memory
    if (globalIdx < n) {
        smem[localIdx] = d_in[globalIdx];
        if (threadIdx.x < RADIUS) { // Load halo elements
            smem[localIdx - RADIUS] = (globalIdx >= RADIUS) ? d_in[globalIdx - RADIUS] : 0.0f; // Left boundary
            smem[localIdx + BLOCK_SIZE] = (globalIdx + BLOCK_SIZE < n) ? d_in[globalIdx + BLOCK_SIZE] : 0.0f; // Right boundary
        }
    }
    __syncthreads(); // Ensure all threads have loaded data

    // Apply stencil operation (simple average)
    if (globalIdx < n && globalIdx >= RADIUS && globalIdx < n - RADIUS) {
        d_out[globalIdx] = 0.25f * (smem[localIdx - 1] + smem[localIdx] + smem[localIdx + 1] + smem[localIdx + 1]);
    }
}

int main() {
    const int n = 1024;  // Number of elements in the array
    float h_in[n], h_out[n];  // Host arrays
    float *d_in, *d_out;  // Device arrays

    // Initialize input array with some values
    for (int i = 0; i < n; i++) {
        h_in[i] = (float)i;
    }

    // Allocate memory on GPU
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    stencil_1d<<<numBlocks, BLOCK_SIZE>>>(d_out, d_in, n);

    // Copy result back to CPU
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print a few results
    printf("Sample output:\n");
    for (int i = 0; i < 10; i++) {
        printf("h_out[%d] = %f\n", i, h_out[i]);
    }

    // Free memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
