#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>


__global__ void invertMatrix(float* d_matrix, float* d_inverse, int N);


int main () {
    int N =3;
    float h_mat[9] = {
        12,5,2,
        6,8,4,
        7,10,4
    }
    int size = N*N*sizeof(float);
    float* d_mat;
    float* d_inv;

    cudaMalloc((void**)&d_mat,size);
    cudaMemcpy(d_mat,h_mat,size,cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_inv,size);

    invertMatrix<<<1,20>>>(d_mat,d_inv,N);

    float* res_mat;

    cudaMemcpy(res_mat,d_inv,size,cudaMemcpyDeviceToHost);

    for (int i=0;i<N;i++) {
        for (int j=0;j<N;j++) {
            cout << res_mat[(i*N)+j] << " ";
        }
        cout << endl;
    }


}

__global__ void invertMatrix(float* d_matrix, float* d_inverse, int N) {
    extern __shared__ float shared[];
    float* matrix = shared;
    float* inverse = shared + N * N;

    int tid = threadIdx.x;

    // Load the matrix and set up the identity matrix in shared memory
    if (tid < N) {
        for (int i = 0; i < N; i++) {
            matrix[tid * N + i] = d_matrix[tid * N + i];
            inverse[tid * N + i] = (tid == i) ? 1.0f : 0.0f;
        }
    }
    __syncthreads();

    // Gaussian elimination to create the inverse
    for (int i = 0; i < N; i++) {
        float pivot = matrix[i * N + i];
        if (pivot == 0) return; // Singular matrix, can't invert

        for (int j = 0; j < N; j++) {
            matrix[i * N + j] /= pivot;
            inverse[i * N + j] /= pivot;
        }

        __syncthreads();

        for (int k = 0; k < N; k++) {
            if (k != i) {
                float factor = matrix[k * N + i];
                for (int j = 0; j < N; j++) {
                    matrix[k * N + j] -= factor * matrix[i * N + j];
                    inverse[k * N + j] -= factor * inverse[i * N + j];
                }
            }
        }
        __syncthreads();
    }

    // Copy the result to the output matrix
    if (tid < N) {
        for (int i = 0; i < N; i++) {
            d_inverse[tid * N + i] = inverse[tid * N + i];
        }
    }
}