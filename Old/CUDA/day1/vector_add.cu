#include <stdio.h>
#include <stdlib.h>

#define N 10000000

__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

int main(){
    float *a, *b, *out; 
    float *d_a, *d_b, *d_out;

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);


    if (a == NULL || b == NULL || out == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Initialize arrays
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);


    // Perform vector addition
    vector_add<<<1,1>>>(d_out, d_a, d_b, N);


    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    printf("\nFirst 10 elements of vectors on GPU:\n");
    for (int i = 0; i < 10; i++) {
        printf("GPU out[%d] = %f (expected: 3.000000)\n", i, out[i]);
    }

    // Free allocated memory
    free(a);
    free(b);
    free(out);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return 0;
}
