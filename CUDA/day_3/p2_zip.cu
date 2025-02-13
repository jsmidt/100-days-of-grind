#include<stdio.h>
#include<stdlib.h>

#define N 4

__global__ void device_zip(int *a, int *b, int *out) {

        int index = threadIdx.x + blockIdx.x * blockDim.x;
        out[index] = a[index] + b[index];
}


int main(void) {
        int *a, *b, *out;
        int *d_a, *d_b, *d_out; // device copies of a, b, c
        int size = N * sizeof(int);

        // Alloc space for host copies of a, b, c and setup input values
        a = (int *)malloc(size);
        b = (int *)malloc(size);
        out = (int *)malloc(size);


        // Alloc space for device copies of a, b, c
        cudaMalloc((void **)&d_a, size);
        cudaMalloc((void **)&d_b, size);
        cudaMalloc((void **)&d_out, size);

        // Replicate arange
        for(int i=0;i<N;i++) {
                a[i] = i;
                b[i] = i;
        }

        // Copy inputs to device
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

        int num_block = 1;
        int threads_per_block = 4;
        device_zip<<<num_block,threads_per_block>>>(d_a,d_b,d_out);

        // Copy result back to host
        cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

        for(int i=0;i<N;i++)
            printf(" %d + %d  = %d\n",  a[i], b[i], out[i]);

        free(a); free(b); free(out);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);

        return 0;
}
