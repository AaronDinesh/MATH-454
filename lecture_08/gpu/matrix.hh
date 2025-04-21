#include <string>
#include <vector>
#include <cuda_runtime.h>

#ifndef __MATRIX_H_
#define __MATRIX_H_

class Matrix {
public:
  double* d_A = NULL;
  Matrix(int m = 0, int n = 0) : m_m(m), m_n(n) {}

  __device__  inline const double &operator()(int i, int j) const { return d_A[i * m_n + j]; }

  inline int m() const { return m_m; }
  inline int n() const { return m_n; }

  __host__ void read(const std::string & filename);
  __host__ ~Matrix();

  //Copies data from the CPU to the GPU
  cudaError_t __host__ copy_to_device(double* &d_a, const std::vector<double>& m_a); 
  
  //Performs the matrix vector multiplication d_y = d_A * d_x
  void __global__ mat_vec_mul(double* d_y, double* d_x, double* d_A, int m, int n);
  //Performs the matrix matrix multiplication d_C = d_A * d_B
  //Matrix A is m x n and matrix B is n x k
  void __global__ mat_mat_mul(double* d_C, double* d_A, double* d_B, int m, int n, int k);

private:
  int m_m{0};
  int m_n{0};
};

#endif // __MATRIX_H_
