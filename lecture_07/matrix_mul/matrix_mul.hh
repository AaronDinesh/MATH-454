#include <cuda_runtime.h>
#include <iostream>
#include <random>

#include "matrix_mul_cpu.hh"
#include "matrix_mul_gpu.hh"

#ifndef MATRIX_MUL_HH
#define MATRIX_MUL_HH

#define CHECK_RESULT 1

inline void printDiff(const matrix_cpu &host_A, const matrix_gpu &device_A) {
  int error_count = 0;
  for (int i = 0; i < host_A.rows(); i++) {
    for (int j = 0; j < host_A.cols(); j++) {
      if (std::abs(host_A(i, j) - device_A(i, j)) > 0.1) {
        std::cout << "diff(" << i << ", " << j << ") CPU=" << host_A(i, j)
                  << ", GPU=" << device_A(i, j) << std::endl;
        error_count++;
      }
    }
  }
  std::cout << "Total Errors = " << error_count << std::endl;
}

// Thread block size
#define BLOCK_SIZE 32

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA (32 * BLOCK_SIZE) // Matrix A width
#define HA (16 * BLOCK_SIZE) // Matrix A height
#define WB (24 * BLOCK_SIZE) // Matrix B width
#define HB WA                // Matrix B height
#define WC WB                // Matrix C width
#define HC HA                // Matrix C height

#endif // MATRIX_MUL_HH
