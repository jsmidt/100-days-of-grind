#include <stdio.h>
#include <stdlib.h>

#define ROWS 2
#define COLS 2
#define N (ROWS * COLS) // Total number of elements

__global__ void device_add(int *a, int *out, int rows, int cols) {
    //printf ("threadIdx.y: %d\n",threadIdx.y);
    //printf ("threadIdx.x: %d\n",threadIdx.x);
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < rows && col < cols) {
        int index = row * cols + col; // Convert 2D index to 1D
        printf ("%d\n",index);
        out[index] = a[index] + 10;
    }
}

int main(void) {
    int *a, *out;
    int *d_a, *d_out; // Device copies of a and out
    int size = N * sizeof(int);

    // Alloc space for host arrays
    a = (int *)malloc(size);
    out = (int *)malloc(size);

    // Initialize 2D array with [[0,1],[2,3]]
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            a[i * COLS + j] = i * COLS + j; // Flatten 2D index
        }
    }

    // Alloc space for device arrays
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_out, size);

    // Copy input to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    // Define CUDA grid and block sizes for 2D processing
    //dim3 threads_per_block(2, 2); // 2x2 threads per block
    //dim3 num_blocks(1, 1);        // 1 block

    // Launch kernel
    int num_blocks = 1;
    //int threads_per_block = 10;
    dim3 threads_per_block(4, 4); // 2x2 threads per block
    device_add<<<num_blocks, threads_per_block>>>(d_a, d_out, ROWS, COLS);

    // Copy result back to host
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("Result (2D Array):\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            int idx = i * COLS + j;
            printf(" %d + 10 = %d  ", a[idx], out[idx]);
        }
        printf("\n");
    }

    // Free memory
    free(a);
    free(out);
    cudaFree(d_a);
    cudaFree(d_out);

    return 0;
}
