#include <iostream>
#include <cuda_runtime.h>

#define SIZE 2  // Define matrix size

// CUDA Kernel
__global__ void call(float *out, float *a, float *b, int size) {
    int local_i = threadIdx.x + blockIdx.x * blockDim.x;
    int local_j = threadIdx.y + blockIdx.y * blockDim.y;

    if (local_i < size && local_j < size) {
        out[local_i * size + local_j] = a[local_i] + b[local_j];
    }
}

int main() {
    // Allocate host memory
    float h_out[SIZE * SIZE] = {0};
    float h_a[SIZE] = {0, 1};  // Example input
    float h_b[SIZE] = {0, 1};  // Example input

    // Allocate device memory
    float *d_out, *d_a, *d_b;
    cudaMalloc((void **)&d_out, SIZE * SIZE * sizeof(float));
    cudaMalloc((void **)&d_a, SIZE * sizeof(float));
    cudaMalloc((void **)&d_b, SIZE * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (SIZE + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    call<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_a, d_b, SIZE);

    // Copy result back to host
    cudaMemcpy(h_out, d_out, SIZE * SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Output Matrix:\n";
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            std::cout << h_out[i * SIZE + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_out);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
