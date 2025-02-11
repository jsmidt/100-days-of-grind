#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vecAdd(float* A, float* B, float* C, int n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n){
        C[i] = A[i] + B[i];
    }
}

int main() {
    int i;
    int n = 10;  // Size of the arrays
    float A_h[10], B_h[10], C_h[10];  // Declare arrays

    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    // Initialize A_h to 1.0 and B_h to 2.0
    for (i = 0; i < n; i++) {
        A_h[i] = 1.0f;
        B_h[i] = 2.0f;
    }

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Call vecAdd
    // vecAdd(A_h, B_h, C_h, n);
    vecAdd<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // Print the results
    printf("C_h array after vecAdd: ");
    for (i = 0; i < n; i++) {
        printf("%.1f ", C_h[i]);
    }
    printf("\n");

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}
