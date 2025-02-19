#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  // Number of elements
#define BLOCK_SIZE 256  // CUDA block size

// CUDA kernel for parallel reduction
__global__ void parallel_sum(float *input, float *output) {
    __shared__ float shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared_data[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // Reduce within the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Write result from block to global memory
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

int main() {
    float h_input[N], h_partial_sums[N / BLOCK_SIZE], sum = 0.0f;
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;  // Fill array with 1s so sum should be N
    }

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, (N / BLOCK_SIZE) * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(N / BLOCK_SIZE);

    parallel_sum<<<gridDim, blockDim>>>(d_input, d_output);
    cudaDeviceSynchronize();

    cudaMemcpy(h_partial_sums, d_output, (N / BLOCK_SIZE) * sizeof(float), cudaMemcpyDeviceToHost);

    // Final summation on CPU
    for (int i = 0; i < N / BLOCK_SIZE; i++) {
        sum += h_partial_sums[i];
    }

    printf("Sum of array: %f\n", sum);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
