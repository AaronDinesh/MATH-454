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

  cudaError_t err = copy_to_device(this->d_a, m_a);
  if (err != cudaSuccess) {
    std::cout << cudaGetErrorString(err) << std::endl;
    if(d_a){
      cudaFree(d_a);
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

__host__ Matrix::~Matrix() {
  if(d_a){
    cudaFree(d_a);
  }
}