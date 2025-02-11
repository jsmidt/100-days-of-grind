#include <stdio.h>

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    int i;
    for (i = 0; i < n; i++) {
        C_h[i] = A_h[i] + B_h[i];
    }
}

int main() {
    int i;
    int n = 10;  // Size of the arrays
    float A_h[10], B_h[10], C_h[10];  // Declare arrays

    // Initialize A_h to 1.0 and B_h to 2.0
    for (i = 0; i < n; i++) {
        A_h[i] = 1.0f;
        B_h[i] = 2.0f;
    }

    // Call vecAdd
    vecAdd(A_h, B_h, C_h, n);

    // Print the results
    printf("C_h array after vecAdd: ");
    for (i = 0; i < n; i++) {
        printf("%.1f ", C_h[i]);
    }
    printf("\n");

    return 0;
}