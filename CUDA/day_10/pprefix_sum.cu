#include <stdio.h>
#include <cuda_runtime.h>

#define N 8  // Number of elements (Must be a power of 2)
#define BLOCK_SIZE 8  // Block size (Same as N for simplicity)

// Kernel for parallel prefix sum (Blelloch Scan)
__global__ void prefix_sum(int *d_input, int *d_output) {
    __shared__ int temp[N];  // Shared memory for scan operation

    int tid = threadIdx.x;

    // Load input into shared memory
    temp[tid] = d_input[tid];
    __syncthreads();

    // **Up-sweep (Reduce Phase)**
    for (int stride = 1; stride < N; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            temp[tid + 2 * stride - 1] += temp[tid + stride - 1];
        }
        __syncthreads();
    }

    // **Down-sweep (Distribution Phase)**
    if (tid == 0) temp[N - 1] = 0;  // Clear last element for exclusive scan
    __syncthreads();

    for (int stride = N / 2; stride > 0; stride /= 2) {
        if (tid % (2 * stride) == 0) {
            int t = temp[tid + stride - 1];
            temp[tid + stride - 1] = temp[tid + 2 * stride - 1];
            temp[tid + 2 * stride - 1] += t;
        }
        __syncthreads();
    }

    // Write result back to output array
    d_output[tid] = temp[tid];
}

int main() {
    int h_input[N] = {1, 2, 3, 4, 5, 6, 7, 8};  // Example input
    int h_output[N];

    int *d_input, *d_output;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, N * sizeof(int));

    // Copy input data to GPU
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    prefix_sum<<<1, BLOCK_SIZE>>>(d_input, d_output);
    cudaDeviceSynchronize();

    // Copy result back to CPU
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    printf("Input:  ");
    for (int i = 0; i < N; i++) printf("%d ", h_input[i]);
    printf("\nOutput: ");
    for (int i = 0; i < N; i++) printf("%d ", h_output[i]);
    printf("\n");

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
