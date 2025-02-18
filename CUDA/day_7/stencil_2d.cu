#include <stdio.h>
#include <cuda_runtime.h>

#define N 10  // Grid size (NxN)
#define BLOCK_SIZE 16  // CUDA block size

// CUDA kernel for 2D stencil
__global__ void stencil_2D(float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Avoid out-of-bounds memory access
    if (i > 0 && i < n-1 && j > 0 && j < n-1) {
        output[j * n + i] = (input[j * n + i] +
                             input[(j-1) * n + i] +
                             input[(j+1) * n + i] +
                             input[j * n + (i-1)] +
                             input[j * n + (i+1)]) / 5.0f;
    }
}

int main() {
    // Host memory allocation
    float h_input[N * N], h_output[N * N];

    // Initialize input grid with some values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_input[i * N + j] = (float)(i * N + j);
        }
    }

    // Device memory allocation
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, N * N * sizeof(float));
    cudaMalloc((void**)&d_output, N * N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    stencil_2D<<<gridDim, blockDim>>>(d_input, d_output, N);
    cudaDeviceSynchronize();  // Wait for GPU to finish execution

    // Copy result back to host
    cudaMemcpy(h_output, d_output, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print part of the output
    printf("Output Grid:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%5.1f ", h_output[i * N + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
