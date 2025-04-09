#include <vector>

#ifndef MATRIX_MUL_CPU_HH
#define MATRIX_MUL_CPU_HH

class matrix_gpu;

class matrix_cpu {
public:
  matrix_cpu(int n, int m) : n(n), m(m), data(n * m) {}
  inline float &operator()(int i, int j) { return data[i * n + j]; }
  inline const float &operator()(int i, int j) const { return data[i * n + j]; }

  int rows() const { return m; }
  int cols() const { return n; }

private:
  int n;
  int m;
  std::vector<float> data;
};

void matMulCPU(matrix_cpu &C, const matrix_gpu &A, const matrix_gpu &B);

#endif /* MATRIX_MUL_CPU_HH */
