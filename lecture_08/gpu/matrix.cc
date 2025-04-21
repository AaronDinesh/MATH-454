#include "matrix.hh"
#include "matrix_coo.hh"
#include <iostream>
#include <string>
#include <cuda_runtime.h>


void __host__ Matrix::read(const std::string & fn) {
  MatrixCOO mat;
  mat.read(fn);
  
  std::vector<double> m_a;
  m_a.resize(mat.m(), mat.n());

  for (int z = 0; z < mat.nz(); ++z) {
    auto i = mat.irn[z];
    auto j = mat.jcn[z];
    auto a = mat.a[z];

    m_a[i * m_n + j] = a;
    if (mat.is_sym()) {
      m_a[j * m_n + i] = a;
    }
  }

  cudaError_t err = copy_to_device(this->d_A, m_a);
  if (err != cudaSuccess) {
    std::cout << cudaGetErrorString(err) << std::endl;
    if(d_A){
      cudaFree(d_A);
    }
    return;
  }  

}

cudaError_t __host__ Matrix::copy_to_device(double* &d_a, const std::vector<double>& m_a) {
  //Malloc memory first
  cudaError_t cudaStatus = cudaMallocManaged((void**)&d_a, sizeof(double) * m_m * m_n);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    return cudaStatus;
  }
  //Copy memory to device
  cudaStatus = cudaMemcpy(d_a, m_a.data(), sizeof(double) * m_m * m_n, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    return cudaStatus;
  }
  return cudaStatus;
}

void __global__ Matrix::mat_vec_mul(double* d_y, double* d_x, double* d_A, int m, int n) {
  int matrix_row = blockIdx.x * blockDim.x + threadIdx.x;

  if (matrix_row < m) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
      sum += d_A[matrix_row * n + i] * d_x[i];
    }
    d_y[matrix_row] = sum;
  }

}

void __global__ Matrix::mat_mat_mul(double* d_C, double* d_A, double* d_B, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < m && col < k){
      double sum = 0.0;
      for(int p = 0; p < n; p++){
        sum += d_A[row * n + p] * d_B[p * k + col];
      }
      d_C[row * k + col] = sum;
    }
}

__host__ Matrix::~Matrix() {
  if(d_A){
    cudaFree(d_A);
  }
}