#include<stdio.h>
#include<stdlib.h>

#define N 4

__global__ void device_add(int *a, int *out) {

        int index = threadIdx.x + blockIdx.x * blockDim.x;
        out[index] = a[index] + 10;
}


int main(void) {
        int *a, *out;
        int *d_a, *d_out; // device copies of a, b, c
        int size = N * sizeof(int);

        // Alloc space for host copies of a, b, c and setup input values
        a = (int *)malloc(size);
        out = (int *)malloc(size);


        // Alloc space for device copies of a, b, c
        cudaMalloc((void **)&d_a, size);
        cudaMalloc((void **)&d_out, size);

        // Replicate arange
        for(int i=0;i<N;i++)
                a[i] = i;

       // Copy inputs to device
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

        int num_block = 1;
        int threads_per_block = 4;
        device_add<<<num_block,threads_per_block>>>(d_a,d_out);

        // Copy result back to host
        cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

        for(int i=0;i<N;i++)
            printf("\n %d + 10  = %d",  a[i], out[i]);

        free(a); free(out);
        cudaFree(d_a); cudaFree(d_out);

        return 0;
}
