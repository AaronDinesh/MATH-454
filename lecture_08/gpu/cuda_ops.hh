#ifndef __CUDA_OPS_H_
#define __CUDA_OPS_H_

#include <cuda_runtime.h>

//Performs elementwise addition of two vectors. Must be spawned with a 2d grid
//and atleast as many threads as elements in the vector (n)
__global__ void vector_add(float* c, const float* a, const float* b, int n);

//Performs the daxpy operation c = a * x + y. Must be spawned with a 2d grid
//and atleast as many threads as elements in the vector (n)
__global__ void cu_daxpy(float* c, const float a, const float* x, const float* y, int n);

//Performs the ddot operation c = x * y. Must be spawned with a 2d grid
//and atleast as many threads as elements in the vector (n)
__global__ void cu_ddot(float* c, const float* x, const float* y, int n);

//Performs the dgemv operation c = alpha * A * x + beta * c. Must be spawned with a 2d grid
//and atleast as many threads as elements in the vector (n)
__global__ void cu_dgemv(float* c, const float* A, const float* x, const float alpha, const float beta, int m, int n);

__global__ void cu_dgemm(float* C, const float* A, const float* B, const float alpha, const float beta, int m, int n, int k);
#endif