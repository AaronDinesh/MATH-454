#include <string>
#include <vector>
#include <cuda_runtime.h>

#ifndef __MATRIX_H_
#define __MATRIX_H_

class Matrix {
public:
  Matrix(int m = 0, int n = 0) : m_m(m), m_n(n) {}

  __device__  inline const double &operator()(int i, int j) const { return d_a[i * m_n + j]; }

  inline int m() const { return m_m; }
  inline int n() const { return m_n; }

  __host__ void read(const std::string & filename);
  __host__ ~Matrix();

  cudaError_t __host__ copy_to_device(double* &d_a, const std::vector<double>& m_a); 


private:
  int m_m{0};
  int m_n{0};
  double* d_a = NULL;
};

#endif // __MATRIX_H_
