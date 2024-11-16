#include <stdio.h>

// Kernel to calculate the inverse of a matrix using Gauss-Jordan elimination and compute the determinant
__global__ void matrixInverse(float *matrices, float *inverses, float *determinants, int n) {
    int idx = blockIdx.x; // Each block handles one matrix

    // Pointers to the current matrix and its inverse
    float *matrix = matrices + idx * n * n;
    float *inverse = inverses + idx * n * n;

    // Initialize the inverse matrix as an identity matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inverse[i * n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    float determinant = 1.0f; // Initialize determinant

    // Perform Gauss-Jordan elimination
    for (int i = 0; i < n; i++) {
        // Make the diagonal element 1 and update the determinant
        float diagElement = matrix[i * n + i];
        if (diagElement == 0) {
            printf("Singular matrix detected\n");
            determinants[idx] = 0.0f; // Matrix is singular, determinant is 0
            return; // Matrix is singular, cannot be inverted
        }
        determinant *= diagElement;

        for (int j = 0; j < n; j++) {
            matrix[i * n + j] /= diagElement;
            inverse[i * n + j] /= diagElement;
        }

        // Eliminate other rows
        for (int k = 0; k < n; k++) {
            if (k != i) {
                float factor = matrix[k * n + i];
                for (int j = 0; j < n; j++) {
                    matrix[k * n + j] -= factor * matrix[i * n + j];
                    inverse[k * n + j] -= factor * inverse[i * n + j];
                }
            }
        }
    }

    determinants[idx] = determinant; // Store the determinant
}

int main() {
    const int n = 3; // Matrix size (example: 3x3)
    const int numMatrices = 1; // Number of matrices to invert
    float h_matrices[numMatrices * n * n] = {
        2, -1, 0,
        -1, 2, -1,
        0, -1, 2
    };

    float h_inverses[numMatrices * n * n];
    float h_determinants[numMatrices];

    float *d_matrices, *d_inverses, *d_determinants;
    cudaMalloc((void **)&d_matrices, numMatrices * n * n * sizeof(float));
    cudaMalloc((void **)&d_inverses, numMatrices * n * n * sizeof(float));
    cudaMalloc((void **)&d_determinants, numMatrices * sizeof(float));

    cudaMemcpy(d_matrices, h_matrices, numMatrices * n * n * sizeof(float), cudaMemcpyHostToDevice);

    matrixInverse<<<numMatrices, 1>>>(d_matrices, d_inverses, d_determinants, n);

    cudaMemcpy(h_inverses, d_inverses, numMatrices * n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_determinants, d_determinants, numMatrices * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Inverted matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", h_inverses[i * n + j]);
        }
        printf("\n");
    }

    printf("Determinant: %f\n", h_determinants[0]);

    cudaFree(d_matrices);
    cudaFree(d_inverses);
    cudaFree(d_determinants);
    return 0;
}
