#include <stdio.h>
#include <cuda_runtime.h>

#define N 10  // Grid size (NxN)
#define BLOCK_SIZE 16  // CUDA block size

// CUDA kernel with shared memory optimization
__global__ void stencil_2D_shared(float *input, float *output, int n) {
    __shared__ float tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int local_i = threadIdx.x + 1;
    int local_j = threadIdx.y + 1;

    // Load global memory into shared memory (including halos)
    if (i < n && j < n) {
        tile[local_j][local_i] = input[j * n + i];
        if (threadIdx.x == 0 && i > 0)
            tile[local_j][0] = input[j * n + (i - 1)];
        if (threadIdx.x == blockDim.x - 1 && i < n - 1)
            tile[local_j][BLOCK_SIZE + 1] = input[j * n + (i + 1)];
        if (threadIdx.y == 0 && j > 0)
            tile[0][local_i] = input[(j - 1) * n + i];
        if (threadIdx.y == blockDim.y - 1 && j < n - 1)
            tile[BLOCK_SIZE + 1][local_i] = input[(j + 1) * n + i];
    }

    __syncthreads();  // Synchronize threads before computation

    // Compute stencil
    if (i > 0 && i < n-1 && j > 0 && j < n-1) {
        output[j * n + i] = (tile[local_j][local_i] +
                             tile[local_j - 1][local_i] +
                             tile[local_j + 1][local_i] +
                             tile[local_j][local_i - 1] +
                             tile[local_j][local_i + 1]) / 5.0f;
    }
}

int main() {
    float h_input[N * N], h_output[N * N];

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            h_input[i * N + j] = (float)(i * N + j);

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, N * N * sizeof(float));
    cudaMalloc((void**)&d_output, N * N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    stencil_2D_shared<<<gridDim, blockDim>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output Grid:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%5.1f ", h_output[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
