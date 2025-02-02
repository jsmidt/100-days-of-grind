#include <stdio.h>
#include <stdlib.h>

#define N 10000000

void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

int main(){
    float *a, *b, *out; 

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    if (a == NULL || b == NULL || out == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Initialize arrays
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Print some values before computation
    printf("Initial values of a and b (first 10 elements):\n");
    for(int i = 0; i < 10; i++) {
        printf("a[%d] = %f, b[%d] = %f\n", i, a[i], i, b[i]);
    }

    // Perform vector addition
    vector_add(out, a, b, N);

    // Print some results after computation
    printf("\nResulting values in out (first 10 elements):\n");
    for(int i = 0; i < 10; i++) {
        printf("out[%d] = %f\n", i, out[i]);
    }

    // Free allocated memory
    free(a);
    free(b);
    free(out);

    return 0;
}
