#ifndef KERNELS_H
#define KERNELS_H

// Declare the CUDA function using `extern "C"` to prevent C++ name mangling
extern "C" void GMM_training(float* pi_k,float* u_k,float* E_k,float* data,int K,int D,int N,float threshold);
extern "C" void GMM_inference(int* labels,float* pi_k,float* u_k,float* E_k,float* data,int K,int D,int N);

#endif // KERNEL_H