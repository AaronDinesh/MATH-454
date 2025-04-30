#include "config.hh"
#include <cuda_runtime.h>
#include "cuda_ops.hh"
#include <iostream>

template <typename T>
__global__ void vector_add(T* c, const T* a, const T* b, size_t n) {
  #if MAX_THREADED_MODE
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      c[idx] = a[idx] + b[idx];
    }
  #else
    size_t total_threads = blockDim.x * gridDim.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = idx; i < n; i += total_threads) {
      c[i] = a[i] + b[i];
    }
  #endif
}

template <typename T>
__global__ void cu_daxpy(const T alpha, const T* x, T* y, size_t n) {
  #if MAX_THREADED_MODE
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      y[idx] = alpha * x[idx] + y[idx];
    }
  #else
    size_t total_threads = blockDim.x * gridDim.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    #pragma unroll 4
    for (size_t i = idx; i < n; i += total_threads) {
      y[i] = alpha * x[i] + y[i];
    }
  #endif
}

template <typename T>
__global__ void cu_daxpy(const T* alpha, const T* x, T* y, size_t n) {
  T alpha_val = *alpha;
  #if MAX_THREADED_MODE
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      y[idx] = alpha_val * x[idx] + y[idx];
    }
  #else
    size_t total_threads = blockDim.x * gridDim.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    #pragma unroll 4
    for (size_t i = idx; i < n; i += total_threads) {
      y[i] = alpha_val * x[i] + y[i];
    }
  #endif
}


template <typename T>
__global__ void cu_ddot(T* c, const T* x, const T* y, size_t n){
  #if MAX_THREADED_MODE
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      T partial_product = x[idx] * y[idx];
      atomicAdd(c, partial_product);
    }
  #else
    size_t total_threads = blockDim.x * gridDim.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll 4
    for (size_t i = idx; i < n; i += total_threads) {
      T partial_product = x[i] * y[i];
      atomicAdd(c, partial_product);
    }
  #endif
}

template <typename T>
__global__ void cu_dgemv(T* c, const  T* A, const T* x, const T alpha, const T beta, size_t m, size_t n){
  #if MAX_THREADED_MODE
    size_t matrix_row = blockIdx.x * blockDim.x + threadIdx.x;

    if (matrix_row < m) {
      T sum = 0;

      #pragma unroll 8
      for (size_t i = 0; i < n; i++) {
        sum += A[matrix_row * n + i] * x[i];
      }
      c[matrix_row] = alpha * sum + beta * c[matrix_row];
    }
  #else
    size_t total_threads = blockDim.x * gridDim.x;
    size_t matrix_row = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t row = matrix_row; row < m; row += total_threads) {
      T sum = 0.0f;

      #pragma unroll 4
      for (size_t i = 0; i < n; ++i) {
        sum += A[row * n + i] * x[i];
      }
      c[row] = alpha * sum + beta * c[row];
    }
  #endif
}

template <typename T>
__global__ void cu_dgemm(T* C, const T* A, const T* B, const T alpha, const T beta, size_t m, size_t n, size_t k){
  #if MAX_THREADED_MODE
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
      T sum = 0.0f;
      
      #pragma unroll 4
      for (size_t i = 0; i < k; ++i) {
        sum += A[row * k + i] * B[i * n + col];
      }
      C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
  #else
    size_t total_threads = blockDim.x * gridDim.x;
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    size_t total_elements = m * n;

    for (size_t idx = thread_id; idx < total_elements; idx += total_threads) {
      size_t row = idx / n;
      size_t 
  col = idx % n;

      T sum = 0.0f;
      #pragma unroll 4
      for (size_t i = 0; i < k; ++i) {
        sum += A[row * k + i] * B[i * n + col];
      }
      C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
  #endif
}


template <typename T>
__global__ void cu_negate(T* a, size_t n){
  if(n == 1){
    if(threadIdx.x == 0 && blockIdx.x == 0){
      *a = -(*a);
    }
    return;
  }
  
  #if MAX_THREADED_MODE
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      a[idx] = -a[idx];
    }
  #else
    size_t total_threads = blockDim.x * gridDim.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
      
    #pragma unroll 4
    for (size_t i = idx; i < n; i += total_threads) {
      a[i] = -a[i];
    }
  #endif
}

template <typename T>
__host__ void copy_from_device(T* &h_a, const T* d_a, size_t count){
  size_t size = count * sizeof(T);

  if (!h_a) {
    h_a = (T*) malloc(size);
  }

  #if ERROR_CHECKING
    if (!h_a) {
        std::cout << "Host malloc failed\n";
        return;
    }
  #endif

  cudaError_t cudaStatus = cudaMemcpy(static_cast<void *>(h_a), static_cast<const void*>(d_a), size, cudaMemcpyDeviceToHost); 
  
  #if ERROR_CHECKING
    if (cudaStatus != cudaSuccess) {
      free(h_a);
      h_a = nullptr;
      std::cout << "copy_from_device failed because cudaMemcpy failed! Cuda error below." << std::endl;
      std::cout << "Cuda error:" << cudaGetErrorString(cudaStatus) << std::endl;
    }
  #endif
}

// template <typename T>
// __host__ void copy_from_device(T* h_a, const T* d_a, size_t count){
//   size_t size = count * sizeof(T);
//   cudaError_t cudaStatus = cudaMemcpy(static_cast<void *>(h_a), static_cast<const void*>(d_a), size, cudaMemcpyDeviceToHost); 
  
//   #if ERROR_CHECKING
//     if (cudaStatus != cudaSuccess) {
//       std::cout << "copy_from_device failed because cudaMemcpy failed! Cuda error below." << std::endl;
//       std::cout << "Cuda error:" << cudaGetErrorString(cudaStatus) << std::endl;
//     }
//   #endif
// }

template <typename T>
__host__ void copy_to_device(T* &d_a, const T* h_a, size_t count){
  size_t size = count * sizeof(T);

  cudaError_t cudaStatus;

  if(!d_a){ 
    cudaStatus = cudaMalloc((void**) &d_a, size);
  
    #if ERROR_CHECKING
      if (cudaStatus != cudaSuccess) {
        std::cout << "copy_to_device failed because cudaMalloc failed! Cuda error below." << std::endl;
        std::cout << "Cuda error:" << cudaGetErrorString(cudaStatus) << std::endl;
      }
    #endif
  }

  cudaStatus = cudaMemcpy((void*) d_a, (const void*) h_a, size, cudaMemcpyHostToDevice);
  
  #if ERROR_CHECKING
    if (cudaStatus != cudaSuccess) {
      cudaFree((void*) d_a);
      d_a = nullptr;
      std::cout << "copy_to_device failed because cudaMemcpy failed! Cuda error below." << std::endl;
      std::cout << "Cuda error:" << cudaGetErrorString(cudaStatus) << std::endl;
    }
  #endif
}

template <typename T>
__host__ void zero_malloc_on_device(T* &d_a, size_t count){
  size_t size = count * sizeof(T);

  cudaError_t cudaStatus = cudaMalloc((void**) &d_a, size);

  #if ERROR_CHECKING
    if (cudaStatus != cudaSuccess){
      std::cout << "zero_malloc_on_device failed because cudaMalloc failed! Cuda error below." << std::endl;
      std::cout << "Cuda error:" << cudaGetErrorString(cudaStatus) << std::endl;
    }
  #endif

  cudaStatus = cudaMemset((void*) d_a, 0, size);

  #if ERROR_CHECKING
    if(cudaStatus != cudaSuccess){
      cudaFree((void*) d_a);
      d_a = nullptr;
      std::cout << "zero_malloc_on_device failed because cudaMemset failed! Cuda error below." << std::endl;
      std::cout << "Cuda error:" << cudaGetErrorString(cudaStatus) << std::endl; 
    }
  #endif
}


template <typename T>
__host__ void assign_on_device(const T* src, T* dst, size_t count){
  size_t size = count * sizeof(T);

  cudaError_t cudaStatus = cudaMemcpy((void*) dst, (const void*) src, size, cudaMemcpyDeviceToDevice);
  
  #if ERROR_CHECKING
    if (cudaStatus != cudaSuccess) {
      std::cout << "assign_on_device failed! Cuda error below." << std::endl;
      std::cout << "Cuda error:" << cudaGetErrorString(cudaStatus) << std::endl;
    }
  #endif
}

template <typename T>
__host__ void assign_on_device(const T a, T* dst){
  cudaError_t cudaStatus = cudaMemcpy((void*) dst, (const void*) &a, sizeof(T), cudaMemcpyHostToDevice);
  
  #if ERROR_CHECKING
    if (cudaStatus != cudaSuccess) {
      std::cout << "assign_on_device failed! Cuda error below." << std::endl;
      std::cout << "Cuda error:" << cudaGetErrorString(cudaStatus) << std::endl;
    }
  #endif
}

template <typename T>
__host__  void zero_fill_array(T* dst, size_t count){
  size_t size = count * sizeof(T);

  cudaError_t cudaStatus = cudaMemset((void*) dst, (T) 0, size);

  #if ERROR_CHECKING
    if (cudaStatus != cudaSuccess) {
      std::cout << "zero_fill_array failed! Cuda error below." << std::endl;
      std::cout << "Cuda error:" << cudaGetErrorString(cudaStatus) << std::endl;
    }
  #endif
}

template <typename T>
__host__ void free_on_device(T* &d_a){
  if(d_a){
    cudaError_t cudaStatus = cudaFree((void*) d_a);
    d_a = nullptr;
    #if ERROR_CHECKING
    if (cudaStatus != cudaSuccess) {
      std::cout << "free_on_device failed! Cuda error below." << std::endl;
      std::cout << "Cuda error:" << cudaGetErrorString(cudaStatus) << std::endl;
    }
    #endif
  }
}

template <typename T>
__global__ void compute_alpha(T* alpha, const T* rsold, const T* d_pAp){
  if(threadIdx.x == 0 && blockIdx.x == 0){
    *alpha = *rsold / fmax(*d_pAp, *rsold * static_cast<T>(1.0e-14));
  }
}

template <typename T>
__global__ void compute_beta(T* beta, const T* rsold, const T* rsnew){
  if(threadIdx.x == 0 && blockIdx.x == 0){
    *beta = *rsnew / *rsold;
  } 
}

// For device functions
template void copy_to_device<double>(double*&, const double*, size_t);
//template void copy_from_device<double>(double*, const double*, size_t);
template void copy_from_device<double>(double*&, const double*, size_t);
template void zero_malloc_on_device<double>(double*&, size_t);
template void assign_on_device<double>(const double*, double*, size_t);
template void assign_on_device<double>(const double, double*);
template void free_on_device<double>(double*&);
template void zero_fill_array<double>(double*, size_t);

// For CUDA kernels (__global__)
template __global__ void cu_daxpy<double>(const double*, const double*, double*, size_t);
template __global__ void cu_daxpy<double>(const double, const double*, double*, size_t);
template __global__ void cu_ddot<double>(double*, const double*, const double*, size_t);
template __global__ void cu_negate<double>(double*, size_t);
template __global__ void cu_dgemv<double>(double*, const double*, const double*, double, double, size_t, size_t);
template __global__ void compute_beta<double>(double*, const double*, const double*);
template __global__ void compute_alpha<double>(double*, const double*, const double*);
