#include "config.h"
#include <cuda_runtime.h>
#include "cuda_ops.hh"
#include <iostream>

__global__ void vector_add(float* c, const float* a, const float* b, int n) {
  #ifdef MAX_THREADED_MODE
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      c[idx] = a[idx] + b[idx];
    }
  #else
    int total_threads = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < n; i += total_threads) {
      c[i] = a[i] + b[i];
    }
  #endif


}

__global__ void cu_daxpy(float* c, const float a, const float* x, const float* y, int n) {
  #ifdef MAX_THREADED_MODE
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      c[idx] = a * x[idx] + y[idx];
    }
  #else
    int total_threads = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < n; i += total_threads) {
      c[i] = a * x[i] + y[i];
    }
  #endif
}

__global__ void cu_ddot(float* c, const float* x, const float* y, int n){
  #ifdef MAX_THREADED_MODE
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      c[idx] = x[idx] * y[idx];
    }
  #else
    int total_threads = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < n; i += total_threads) {
      c[i] = x[i] * y[i];
    }
  #endif
}

__global__ void cu_dgemv(float* c, const float* A, const float* x, const float alpha, const float beta, int m, int n){
  #ifdef MAX_THREADED_MODE
    int matrix_row = blockIdx.x * blockDim.x + threadIdx.x;

    if (matrix_row < m) {
      double sum = 0;
      for (int i = 0; i < n; i++) {
        sum += A[matrix_row * n + i] * x[i];
      }
      c[matrix_row] = alpha * sum + beta * c[matrix_row];
    }
  #else
    int total_threads = blockDim.x * gridDim.x;
    int matrix_row = blockIdx.x * blockDim.x + threadIdx.x;

    for (int row = matrix_row; row < m; row += total_threads) {
      float sum = 0.0f;
      for (int i = 0; i < n; ++i) {
        sum += A[row * n + i] * x[i];
      }
      c[row] = alpha * sum + beta * c[row];
    }
  #endif
}

__global__ void cu_dgemm(float* C, const float* A, const float* B, const float alpha, const float beta, int m, int n, int k){
  #ifdef MAX_THREADED_MODE
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
      float sum = 0.0f;
      for (int i = 0; i < k; ++i) {
        sum += A[row * k + i] * B[i * n + col];
      }
      C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
  #else
    int total_threads = blockDim.x * gridDim.x;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    int total_elements = m * n;

    for (int idx = thread_id; idx < total_elements; idx += total_threads) {
      int row = idx / n;
      int col = idx % n;

      float sum = 0.0f;
      for (int i = 0; i < k; ++i) {
        sum += A[row * k + i] * B[i * n + col];
      }
      C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
  #endif
}

template <typename T>
__host__ void copy_from_device(T* &h_a, const T* d_a, size_t count){
  size_t size = count * sizeof(T);
  cudaError_t cudaStatus = cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost); 
  if (cudaStatus != cudaSuccess) {
    free(h_a);
    h_a = nullptr;
    std::cout << "Cuda error:" << cudaGetErrorString(cudaStatus) << std::endl;
  }
}