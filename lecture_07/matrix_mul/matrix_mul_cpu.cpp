#include "matrix_mul.hh"
#include <iostream>

void matMulCPU(matrix_cpu &C, const matrix_gpu &A, const matrix_gpu &B) {
  for (int i = 0; i < C.rows(); ++i)
    for (int j = 0; j < C.cols(); ++j) {
      float sum = 0;
      for (int k = 0; k < A.cols(); ++k) {
        float a = A(i, k);
        float b = B(k, j);
        sum += a * b;
      }
      C(i, j) = sum;
    }
}
