#include "matrix_mul.hh"

__global__ void matMulGPU_naive(matrix_gpu C, const matrix_gpu A,
                                const matrix_gpu B) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  float sum = 0.0;

  for (int k = 0; k < A.cols(); k++) {
    sum += A(i, k) * B(k, j);
  }

  C(i, j) = sum;
}
