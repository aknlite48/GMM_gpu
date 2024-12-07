#include <cuda_runtime.h>
#include <iostream>
#include "kernels.h" 

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
	float temp_sum=0;
	for (int tid=tid0;tid<N;tid+=tot_threads) {
		temp_sum+=resp[(tid*K)+bid];
		//atomicAdd(&sum,resp[(tid*K)+bid]);
	}
	atomicAdd(&sum,temp_sum);
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


	for (int i=0;i<D;i++) {
		float temp_sum=0;
		for (int tid=tid0;tid<N;tid+=tot_threads) {
			temp_sum+=resp[(tid*K)+bid]*data[(tid*D)+i];
	}
		atomicAdd(&shared_val[i],temp_sum);
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
                for (int i=0;i<size;i++) {
                        int row = i/D;
                        int col = i%D;
			float temp_sum=0;
for (int tid=tid0;tid<N;tid+=tot_threads) {
temp_sum+=(resp[(tid*K)+bid])*(data[(tid*D)+row]-u_k[(bid*D)+row])*(data[(tid*D)+col]-u_k[(bid*D)+col]);
}
atomicAdd(&shared_mat[i],temp_sum);
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

__global__ void add_label(float* resp,int* labels,int N,int K) {
	//multiblock
	int tid0 = (blockIdx.x*blockDim.x)+threadIdx.x;
	int tot_threads = gridDim.x*blockDim.x;

	for (int tid=tid0;tid<N;tid+=tot_threads) {
		float cmax = resp[(tid*K)]; int maxi=0;
		for (int i=1;i<K;i++) {
			float val = resp[(tid*K)+i];
			if (val>cmax) {
				cmax = val; maxi=i;
			}
		}
		labels[tid] = maxi;
	}
}

extern "C" int GMM_training(float* pi_k,float* u_k,float* E_k,float* data,int K,int D,int N,float threshold,bool verbose,int max_iter) {

	float log_LL=0; float prev_log_LL=0;

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
	cudaMalloc((void**)&d_n_k,alloc_size1);

	cudaMalloc((void**)&d_log_LL,sizeof(float));


	//cuda memcpy
	cudaMemcpy(d_data,data,N*D*sizeof(float),cudaMemcpyHostToDevice);

	cudaMemcpy(d_pi_k,pi_k,alloc_size1,cudaMemcpyHostToDevice);
	cudaMemcpy(d_u_k,u_k,alloc_size2,cudaMemcpyHostToDevice);
	cudaMemcpy(d_E_k,E_k,alloc_size3,cudaMemcpyHostToDevice);
	
	int calc_blocks = 20;
	int M_step_threads = 300;
	if (N>=1e5) {
		calc_blocks = 60;//M_step_threads=500;
	}

//	int max_iter = 500; 
	int iters=0; if (verbose) {printf("\n");}
	for (int i=0;i<max_iter;i++) {
		iters++;
		//kernel invocation
		//calculate E matrix inverse and determinant
		matrixInverse<<<K,1>>>(d_E_k,d_E_k_inv,d_E_k_det,D);
		cudaDeviceSynchronize();

		//M Step : calc gaussians & update responsibilities
		gauss_calc<<<calc_blocks,500>>>(d_gaussians,d_data,d_u_k,d_E_k_inv,d_E_k_det,d_pi_val,N,K,D);
		cudaDeviceSynchronize();

		cudaMemset(d_log_LL, 0, sizeof(float));
		resp_denom_update<<<calc_blocks,500>>>(d_gaussians,d_pi_k,d_resp_denom,d_log_LL,N,K);
		cudaDeviceSynchronize();

		E_step<<<calc_blocks,500>>>(d_gaussians,d_pi_k,d_resp,d_resp_denom,N,K);
		cudaDeviceSynchronize();

		M_step_nk_update<<<K,M_step_threads>>>(d_resp,d_data,d_n_k,d_u_k,N,K,D);
		cudaDeviceSynchronize();

		M_step_uk_update<<<K,M_step_threads>>>(d_resp,d_data,d_n_k,d_u_k,N,K,D);
		cudaDeviceSynchronize();

		M_step_Ek_update<<<K,M_step_threads>>>(d_resp,d_data,d_u_k,d_E_k,d_n_k,N,K,D);
		pi_k_update<<<1,K>>>(d_pi_k,d_n_k,K,N);
		cudaDeviceSynchronize();

		cudaMemcpy(&log_LL,d_log_LL,sizeof(float),cudaMemcpyDeviceToHost);

		//compare logs
	//	if (verbose) {printf("LL: %f iter: %d \n",log_LL,i);}
	if (verbose) {std::cout << "\r\033[F" << "Iterations: " << i+1 << " LL: " << log_LL <<"\n"<< std::flush;}
		if (abs(log_LL-prev_log_LL)<threshold) {
			break;
		}
		prev_log_LL = log_LL;
	}
    cudaMemcpy(pi_k,d_pi_k,alloc_size1,cudaMemcpyDeviceToHost);
    cudaMemcpy(u_k,d_u_k,alloc_size2,cudaMemcpyDeviceToHost);
    cudaMemcpy(E_k,d_E_k,alloc_size3,cudaMemcpyDeviceToHost);

    cudaFree(d_pi_k);
    cudaFree(d_u_k);
    cudaFree(d_E_k);
    cudaFree(d_E_k_inv);
	cudaFree(d_E_k_det);
	cudaFree(d_gaussians);
	cudaFree(d_resp);
	cudaFree(d_resp_denom);
	cudaFree(d_log_LL);
	cudaFree(d_n_k);
	cudaFree(d_data);
	return iters;
}

extern "C" void GMM_inference(int* labels,float* pi_k,float* u_k,float* E_k,float* data,int K,int D,int N) {
	
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

	int* d_labels;

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
	cudaMalloc((void**)&d_n_k,alloc_size1);

	cudaMalloc((void**)&d_log_LL,sizeof(float));
	cudaMalloc((void**)&d_labels,N*sizeof(int));

	//cuda memcpy
	cudaMemcpy(d_data,data,N*D*sizeof(float),cudaMemcpyHostToDevice);

	cudaMemcpy(d_pi_k,pi_k,alloc_size1,cudaMemcpyHostToDevice);
	cudaMemcpy(d_u_k,u_k,alloc_size2,cudaMemcpyHostToDevice);
	cudaMemcpy(d_E_k,E_k,alloc_size3,cudaMemcpyHostToDevice);



    //setting up inference:
	matrixInverse<<<K,1>>>(d_E_k,d_E_k_inv,d_E_k_det,D);
	cudaDeviceSynchronize();

	//M Step : calc gaussians & update responsibilities
	gauss_calc<<<50,500>>>(d_gaussians,d_data,d_u_k,d_E_k_inv,d_E_k_det,d_pi_val,N,K,D);
	cudaDeviceSynchronize();

	cudaMemset(d_log_LL, 0, sizeof(float));
	resp_denom_update<<<50,500>>>(d_gaussians,d_pi_k,d_resp_denom,d_log_LL,N,K);
	cudaDeviceSynchronize();

	E_step<<<50,500>>>(d_gaussians,d_pi_k,d_resp,d_resp_denom,N,K);
	cudaDeviceSynchronize();

	add_label<<<50,500>>>(d_resp,d_labels,N,K);
	cudaDeviceSynchronize();

	cudaMemcpy(labels,d_labels,N*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(d_pi_k);
    cudaFree(d_u_k);
    cudaFree(d_E_k);
    cudaFree(d_E_k_inv);
        cudaFree(d_E_k_det);
        cudaFree(d_gaussians);
        cudaFree(d_resp);
        cudaFree(d_resp_denom);
        cudaFree(d_log_LL);
        cudaFree(d_n_k);
        cudaFree(d_data);
	cudaFree(d_labels);
}
