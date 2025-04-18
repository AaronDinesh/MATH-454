#include "matrix_mul.hh"

__global__ void matMulGPU_naive(matrix_gpu C, const matrix_gpu A, const matrix_gpu B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < A.rows() && col < B.cols()){
        for(int k = 0; k < A.cols(); k++){
            C(row, col) += A(row, k) * B (k, col);
        }
    }
}
