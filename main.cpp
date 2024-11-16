#include <iostream>
#include <vector>

using namespace std;

#include "csv_read.hpp"

//initial params means(u_k),variances (E_k) and mixing(pi_k)
//D dim data | N points | K clusters

/*
Params
pi_k = [*]xK
u_k  = [*,*]x(K,D)      | access: (i*D)+j
E_k  = [*,*,*]x(K,D,D)  | access: k*(D*D) +(i*D) + j
gaussians = [*,*]x(N,K) | access: i*K + j

resp = [*,*]x(N,K)      | access: i*K + j

E_k_inv = [*,*,*]x(K,D,D)  | access: k*(D*D) +(i*D) + j

Data point:
Dataset = [*,*]x(N,D)   | access: i*(D) + j


*/


//kernels
__global__ void matrixInverse(float *matrices, float *inverses, float *determinants, int n);
__global__ void gauss_calc(float* d_gaussians,float* data,float* u_k,float* E_k_inv,float* E_k_det,int N,int K, int D);

//__global__ void E_step_denom()

//__global__ void E_step





int main() {

	int K=5; int D=2; int N=200;
	float pi_k[K];
	float u_k[(K*D)];
	float E_k[(K*D*D)];
	float data[(N*D)];

	read_data(data,"data.csv");
	read_data(pi_k,"weights.csv");
	read_data(u_k,"means.csv");
	read_data(E_k,"covariances.csv");

	float* d_data;

	float* d_pi_k;
	float* d_u_k;
	float* d_E_k;
	float* d_E_k_inv;
	float* d_E_k_det;

	float* d_gaussians;

	//cuda alloc
	int alloc_size1 = K*sizeof(float);
	int alloc_size2 = K*D*sizeof(float);
	int alloc_size3 = K*D*D*sizeof(float);

	cudaMalloc((void**)&d_data,D*N*sizeof(float));

	cudaMalloc((void**)&d_pi_k,alloc_size1);
	cudaMalloc((void**)&d_u_k,alloc_size2);
	cudaMalloc((void**)&d_E_k,alloc_size3);
	cudaMalloc((void**)&d_E_k_inv,alloc_size3);
	cudaMalloc((void**)&d_E_k_det,alloc_size1);

	//cuda memcpy
	cudaMemcpy(d_pi_k,pi_k,alloc_size1,cudaMemcpyHostToDevice);
	cudaMemcpy(d_u_k,u_k,alloc_size2,cudaMemcpyHostToDevice);
	cudaMemcpy(d_E_k,E_k,alloc_size3,cudaMemcpyHostToDevice);


	//kernel invocation
	//calculate E matrix inverse and determinant
	matrixInverse<<<K,1>>>(d_E_k,d_E_k_inv,d_E_k_det,N);

	//calculate gaussians 


}



//kernel code:
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

    float determinant = 1; // Initialize determinant

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

/*
__global__ void gauss_calc(float* d_gaussians,float* data,float* u_k,float* E_k_inv,float* E_k_det,int N,int K,int D) {

	int tid =;
	int size = N*K;
	if (tid<size) {
		int x_n = tid/K; //xn = d_data[(x_n*D):(x_n)]
		int k_n = tid%K;


		d_gaussians[tid] = gaussian();
	}



}
*/