#include <iostream>
#include <vector>
#include <cuda.h>

# define M_PI 3.14159265358979323846  /* pi */

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

    for (int i = 0; i < n; i++) {
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

    determinants[idx] = fabs(determinant); // abs of det : fix 4
}


__global__ void gauss_calc(float* d_gaussians,float* data,float* u_k,float* E_k_inv,float* E_k_det,float* pi_val,int N,int K,int D) {
	//multi block

	int tid0 = (blockIdx.x*blockDim.x)+threadIdx.x;
	int tot_threads = gridDim.x*blockDim.x;
	int size = N*K;
	for (int tid=tid0;tid<size;tid+=tot_threads) {
		int x_n = tid/K; //xn = d_data[(x_n*D):(x_n)]
		int k_n = tid%K;

		float x_minus_u[10];
		for (int i=0;i<D;i++) {
			x_minus_u[i] = data[(x_n*D)+i]-u_k[k_n*D+i];
		}

		float final_val=0;

		for (int i=0;i<D;i++) {
			float temp_val=0;
			for (int j=0;j<D;j++) {
				temp_val+=E_k_inv[(D*D*k_n)+(D*i)+j]*x_minus_u[j]; //access mistake : fix 1
			}
			final_val+=x_minus_u[i]*temp_val;
		}

		//gaussian function
		d_gaussians[tid] =  exp((-final_val)/2)*rsqrt((*pi_val)*E_k_det[k_n]); //exp((-final_val)/2);
	}
}



__global__ void resp_denom_update(float* gaussians,float* pi_k,float* resp_denom,float* log_LL,int N,int K) {
	//multi block

	int tid0 = (blockIdx.x*blockDim.x)+threadIdx.x;
	int tot_threads = gridDim.x*blockDim.x;
	int size = N;
	for (int tid=tid0;tid<size;tid+=tot_threads) {
		float sum=0;
		for (int i=0;i<K;i++) {
			sum+=gaussians[(tid*K)+i]*pi_k[i];
		}
		resp_denom[tid]=sum;
		atomicAdd(log_LL,log(sum));  //added atomic add : fix 2
	}

}

__global__ void E_step(float* gaussians,float* pi_k,float* resp,float* resp_denom,int N,int K) {
	//multi block

	int tid0 = (blockIdx.x*blockDim.x)+threadIdx.x;
	int tot_threads = gridDim.x*blockDim.x;
	int size = N*K;
	for (int tid=tid0;tid<size;tid+=tot_threads) {
		int x_n = tid/K; //xn = d_data[(x_n*D):(x_n)]
		int k_n = tid%K;
		resp[tid] =  pi_k[k_n]*gaussians[tid]/resp_denom[x_n]; //update rule not included pi: fix 3
	}
}

__global__ void M_step_nk_update(float* resp,float* data,float* n_k,float *u_k,int N,int K,int D) {
	//multi block
	int bid = blockIdx.x;
	int tid0 = threadIdx.x;
	int tot_threads=blockDim.x;

	__shared__ float sum;
	sum=0;
	__syncthreads();

	//Nk calc
	for (int tid=tid0;tid<N;tid+=tot_threads) {

		atomicAdd(&sum,resp[(tid*K)+bid]);
	}
	__syncthreads();

	if (tid0==0) {
		n_k[bid] = sum;
	}
	
	
}

__global__ void M_step_uk_update(float* resp,float* data,float* n_k,float *u_k,int N,int K,int D) {
	int bid = blockIdx.x;
	int tid0 = threadIdx.x;
	int tot_threads = blockDim.x;

	__shared__ float shared_val[10];
	for (int i=0;i<D;i++) {
		shared_val[i]=0;
	}
	__syncthreads();



	for (int tid=tid0;tid<N;tid+=tot_threads) {
		for (int i=0;i<D;i++) {
			atomicAdd(&shared_val[i],resp[(tid*K)+bid]*data[(tid*D)+i]);
		}
	}
	__syncthreads();

	//atomic add:
	if (tid0==0) {
		for (int i=0;i<D;i++) {
			u_k[(bid*D)+i] = shared_val[i]/n_k[bid];
		}
	}
}

__global__ void M_step_Ek_update(float* resp,float* data,float* u_k,float* E_k,float* n_k,int N,int K,int D) {
	//split block for each E_k
	int bid = blockIdx.x;
	int tid0 = threadIdx.x;
	int tot_threads = blockDim.x;  //only tot threads per block

	int size = D*D;


	__shared__ float shared_mat[100]; 

	for (int i=0;i<size;i++) {
		shared_mat[i]=0;
	}
	__syncthreads();

	for (int tid=tid0;tid<N;tid+=tot_threads) {
		for (int i=0;i<size;i++) {
			int row = i/D;
			int col = i%D;
			atomicAdd(&shared_mat[i],(resp[(tid*K)+bid])*(data[(tid*D)+row]-u_k[(bid*D)+row])*(data[(tid*D)+col]-u_k[(bid*D)+col]));
		}
	}

	__syncthreads();

	if (tid0==0) {
		for (int i=0;i<size;i++) {
			E_k[(size*bid)+i] = shared_mat[i]/n_k[bid];
		}
	}
	
}

__global__ void pi_k_update(float* pi_k,float* n_k,int K,int N) {
	//single block
	int tid=threadIdx.x;
	if (tid<K) {
		pi_k[tid] = n_k[tid]/N;
	}
}
int main(int argc,char* argv[]) {
	if (argc!=9) {
		cout << "Incorrect Usage | -K num -D num -N num" << endl;
		return 1;
	}

	int K, D,  N; float threshold;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];

        if (arg == "-N") {
            if (i + 1 < argc) { // Check if there is a value after the flag
                N = std::stoi(argv[++i]); // Parse the next argument as an integer

            } 
        } else if (arg == "-K") {
                        if (i + 1 < argc) { // Check if there is a value after the flag
                K = std::stoi(argv[++i]); // Parse the next argument as an integer

            } 
        } else if (arg=="-D") {
        	            if (i + 1 < argc) { // Check if there is a value after the flag
                D = std::stoi(argv[++i]); // Parse the next argument as an integer

            } 
        } else if (arg=="-T") {
                            if (i + 1 < argc) { // Check if there is a value after the flag
                threshold = std::stof(argv[++i]); // Parse the next argument as an integer

            }
} 
    }
	float pi_k[K];
	float u_k[(K*D)];
	float E_k[(K*D*D)];
	float data[(N*D)];


	float log_LL=0; float prev_log_LL=0;

	read_data(data,"data.csv");
	read_data(pi_k,"weights.csv");
	read_data(u_k,"means.csv");
	read_data(E_k,"covariances.csv");
/*
	//before gpu data check:
	printf("before:");
		printf("\n");
        printf("weights \n");
        for (int i=0;i<K;i++) {
                cout << pi_k[i] << " ";
        }
        printf("\n");
    printf("means: \n");	
	for (int i=0;i<K;i++) {
		for (int j=0;j<D;j++) {
			cout << u_k[(i*D)+j] << " ";
		}
	printf("\n");
	}
	printf("\n");
	printf("covariances: \n");
	for (int k=0;k<K;k++) {
		for (int i=0;i<D;i++) {
			for (int j=0;j<D;j++) {
				cout << E_k[(D*D*k)+(i*D)+j] << " ";
			}
		printf("\n");
		}
		printf("\n");
	}
*/
	float* d_data;
	float pi_val_calc = 2*M_PI; for (int i=1;i<D;i++) {pi_val_calc*=2*M_PI;}
	float* d_pi_val;
       	cudaMalloc((void**)&d_pi_val,sizeof(float)); 
	cudaMemcpy(d_pi_val,&pi_val_calc, sizeof(float),cudaMemcpyHostToDevice);

	float* d_pi_k;
	float* d_u_k;
	float* d_E_k;
	float* d_E_k_inv;
	float* d_E_k_det;

	float* d_gaussians;
	float* d_resp;
	float* d_resp_denom;
	float* d_n_k;
	float* d_log_LL;

	//cuda alloc
	int alloc_size1 = K*sizeof(float);
	int alloc_size2 = K*D*sizeof(float);
	int alloc_size3 = K*D*D*sizeof(float);
	int alloc_size4 = N*K*sizeof(float);

	cudaMalloc((void**)&d_data,D*N*sizeof(float));

	cudaMalloc((void**)&d_pi_k,alloc_size1);
	cudaMalloc((void**)&d_u_k,alloc_size2);
	cudaMalloc((void**)&d_E_k,alloc_size3);
	cudaMalloc((void**)&d_E_k_inv,alloc_size3);
	cudaMalloc((void**)&d_E_k_det,alloc_size1);

	cudaMalloc((void**)&d_gaussians,alloc_size4);
	cudaMalloc((void**)&d_resp,alloc_size4);
	cudaMalloc((void**)&d_resp_denom,N*sizeof(float));
	cudaMalloc((void**)&d_log_LL,sizeof(float));
	cudaMalloc((void**)&d_n_k,alloc_size1);


	//cuda memcpy
	cudaMemcpy(d_data,data,N*D*sizeof(float),cudaMemcpyHostToDevice);

	cudaMemcpy(d_pi_k,pi_k,alloc_size1,cudaMemcpyHostToDevice);
	cudaMemcpy(d_u_k,u_k,alloc_size2,cudaMemcpyHostToDevice);
	cudaMemcpy(d_E_k,E_k,alloc_size3,cudaMemcpyHostToDevice);

	cudaMemcpy(d_log_LL,&log_LL,sizeof(float),cudaMemcpyHostToDevice);

int max_iter = 500;
for (int i=0;i<max_iter;i++) {
	//kernel invocation
	//calculate E matrix inverse and determinant
	matrixInverse<<<K,1>>>(d_E_k,d_E_k_inv,d_E_k_det,D);
	cudaDeviceSynchronize();

	//M Step : calc gaussians & update responsibilities
	gauss_calc<<<20,250>>>(d_gaussians,d_data,d_u_k,d_E_k_inv,d_E_k_det,d_pi_val,N,K,D);
	cudaDeviceSynchronize();

	cudaMemset(d_log_LL, 0, sizeof(float));
	resp_denom_update<<<20,250>>>(d_gaussians,d_pi_k,d_resp_denom,d_log_LL,N,K);
	cudaDeviceSynchronize();

	E_step<<<20,250>>>(d_gaussians,d_pi_k,d_resp,d_resp_denom,N,K);
	cudaDeviceSynchronize();

	M_step_nk_update<<<K,250>>>(d_resp,d_data,d_n_k,d_u_k,N,K,D);
	cudaDeviceSynchronize();

	M_step_uk_update<<<K,250>>>(d_resp,d_data,d_n_k,d_u_k,N,K,D);
	cudaDeviceSynchronize();

	M_step_Ek_update<<<K,250>>>(d_resp,d_data,d_u_k,d_E_k,d_n_k,N,K,D);
	pi_k_update<<<1,K>>>(d_pi_k,d_n_k,K,N);
	cudaDeviceSynchronize();

	cudaMemcpy(&log_LL,d_log_LL,sizeof(float),cudaMemcpyDeviceToHost);

	//compare logs
	printf("LL: %f iter: %d \n",log_LL,i);
	if (abs(log_LL-prev_log_LL)<threshold) {
		break;
	}
	prev_log_LL = log_LL;
//	cout << log_LL << endl;
}
        cudaMemcpy(pi_k,d_pi_k,alloc_size1,cudaMemcpyDeviceToHost);
        cudaMemcpy(u_k,d_u_k,alloc_size2,cudaMemcpyDeviceToHost);
        cudaMemcpy(E_k,d_E_k,alloc_size3,cudaMemcpyDeviceToHost);
}



//kernel code:

