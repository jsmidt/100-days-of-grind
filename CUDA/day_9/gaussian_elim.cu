#include <stdio.h>
#include <cuda_runtime.h>

#define N 4  // Matrix size (4x4)

// Kernel to perform Gaussian elimination
__global__ void gaussian_elimination(float *A, float *b) {
    __shared__ float A_shared[N][N]; // Shared memory for matrix A
    __shared__ float b_shared[N];    // Shared memory for vector b

    int tx = threadIdx.x; // Each thread corresponds to a row

    // Load A and b into shared memory
    for (int i = 0; i < N; i++) {
        A_shared[tx][i] = A[tx * N + i];
    }
    b_shared[tx] = b[tx];

    __syncthreads();

    // Gaussian elimination
    for (int k = 0; k < N; k++) {
        // Pivot: Normalize row k
        float pivot = A_shared[k][k];
        if (tx == k) {
            for (int j = k; j < N; j++) {
                A_shared[k][j] /= pivot;
            }
            b_shared[k] /= pivot;
        }
        __syncthreads();

        // Eliminate other rows
        if (tx > k) {
            float factor = A_shared[tx][k];
            for (int j = k; j < N; j++) {
                A_shared[tx][j] -= factor * A_shared[k][j];
            }
            b_shared[tx] -= factor * b_shared[k];
        }
        __syncthreads();
    }

    // Back-substitution (only performed by one thread)
    if (tx == 0) {
        for (int i = N - 1; i >= 0; i--) {
            for (int j = i + 1; j < N; j++) {
                b_shared[i] -= A_shared[i][j] * b_shared[j];
            }
        }
    }

    __syncthreads();

    // Write back result (solution vector x)
    b[tx] = b_shared[tx];
}

// CPU function to print a 4x4 matrix
void print_matrix(float *A) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%8.3f ", A[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// CPU function to print a vector
void print_vector(float *b) {
    for (int i = 0; i < N; i++) {
        printf("%8.3f\n", b[i]);
    }
    printf("\n");
}

int main() {
    float h_A[N * N] = { // Example 4x4 matrix
        2, -1,  1,  3,
        4,  1, -3,  2,
        -2, 2,  1, -1,
        1,  3, -2,  4
    };

    float h_b[N] = {5, -1, 3, 7};  // Example right-hand side vector

    float *d_A, *d_b;

    // Allocate device memory
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));

    // Copy matrix A and vector b to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel with 4 threads (one per row)
    gaussian_elimination<<<1, N>>>(d_A, d_b);
    cudaDeviceSynchronize();

    // Copy result (vector x) back to host
    cudaMemcpy(h_b, d_b, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print solution
    printf("Solution vector x:\n");
    print_vector(h_b);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_b);

    return 0;
}
