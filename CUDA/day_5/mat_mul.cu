#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16  // Block size for tiling (tune for performance)

// CUDA kernel for matrix multiplication
__global__ void matrixMulCUDA(float *C, float *A, float *B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Host function to perform matrix multiplication on the GPU
void matrixMultiplicationCUDA(float *h_A, float *h_B, float *h_C, int N) {
    float *d_A, *d_B, *d_C;
    int size = N * N * sizeof(float);

    // Allocate device memory
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy input matrices to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    matrixMulCUDA<<<gridDim, blockDim>>>(d_C, d_A, d_B, N);

    // Copy result matrix back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Utility function to print a matrix
void printMatrix(float *M, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%0.2f ", M[i * N + j]);
        }
        printf("\n");
    }
}

// Main function
int main() {
    int N = 4;  // Matrix size (NxN)
    float h_A[N * N], h_B[N * N], h_C[N * N];

    // Initialize matrices A and B with random values
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)(rand() % 10);
        h_B[i] = (float)(rand() % 10);
    }

    printf("Matrix A:\n");
    printMatrix(h_A, N);
    
    printf("\nMatrix B:\n");
    printMatrix(h_B, N);

    // Perform matrix multiplication on GPU
    matrixMultiplicationCUDA(h_A, h_B, h_C, N);

    // Print result
    printf("\nMatrix C (Result):\n");
    printMatrix(h_C, N);

    return 0;
}
