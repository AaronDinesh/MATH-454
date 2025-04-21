#include <cuda_runtime.h>
#include <random>
#include <vector>
#include <iostream>
#include <exception>

/**
 * TODO: write a kernel that does the vector addition C = A + B with 1 thread
 */
__global__ void vectorAddOneThread(const float *A, const float *B, float *C, int N) {
  for(int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
  }
}

/**
 * TODO: write a kernel that does the vector addition C = A + B with 1 Block
 * and 256 threads Hint: When 256 threads are working on one loop how the loop
 * changes?
 */
__global__ void vectorAddOneBlock(const float *A, const float *B, float *C, int N) {
  for(int i = 0; i < N / blockDim.x; i++) {
    int idx = i * blockDim.x + threadIdx.x;
    C[idx] = A[idx] + B[idx];
  }
}

/**
 * TODO: write a kernel that does the vector addition C = A+B with grid of
 * blocks. Each block has 256 threads. Hint: what check do you need to implement
 * to avoid invalid memory reference?
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}

/* -------------------------------------------------------------------------- */
void checkResults(std::string test, const float *A, const float *B,
                  const float *C, int N) {
  // Verify that the result vector is correct
  for (int i = 0; i < N; ++i) {
    if (std::abs(A[i] + B[i] - C[i]) > 1e-5) {
      throw std::runtime_error("Result verification failed at element "
                               + std::to_string(i) + " for test " + test);
    }
  }
}

/**
 * Host main routine
 */
int main() {
  // Print the vector length to be used, and compute its size
  int N = 1 << 20; // 1M elements
  size_t size_in_bytes = N * sizeof(float);
  std::cout << "[Vector addition of " << N << " elements]" << std::endl;

  float *d_A{nullptr};
  float *d_B{nullptr};
  float *d_C{nullptr};

  // TODO: allocate d_A, d_B, and d_C
  cudaMallocManaged(&d_A, size_in_bytes);
  cudaMallocManaged(&d_B, size_in_bytes);
  cudaMallocManaged(&d_C, size_in_bytes);

  std::mt19937 gen(2006);
  std::uniform_real_distribution<> dis(0.f, 1.f);

  // Initialize the input vectors
  for (int i = 0; i < N; ++i) {
    d_A[i] = dis(gen);
    d_B[i] = dis(gen);
  }

  // Launch the Vector Add CUDA Kernel
  int threads_per_block = 256;

  // TODO: Launch the Vector Add CUDA Kernel with one threads
  vectorAddOneThread <<<1, 1>>> (d_A, d_B, d_C, N);
  cudaDeviceSynchronize(); // Since kernel launches is async wrt to the host we
                           // have to syncronize

  checkResults("vectorAddOneThread", d_A, d_B, d_C, N);

  // TODO: Launch the Vector Add CUDA Kernel with one block and 256 threads
  vectorAddOneBlock <<<1, threads_per_block>>> (d_A, d_B, d_C, N);
  cudaDeviceSynchronize();

  checkResults("vectorAddOneBlock", d_A, d_B, d_C, N);

  int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;; // TODO: compute the blocks per grid
  // TODO: Launch the Vector Add CUDA Kernel with blocksPerGrid and 256 threads
  vectorAdd <<<blocks_per_grid, threads_per_block>>> (d_A, d_B, d_C, N);
  cudaDeviceSynchronize();

  checkResults("vectorAdd", d_A, d_B, d_C, N);

  std::cout << "Test PASSED" << std::endl;

  // TODO: Free device global memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  std::cout << "Done" << std::endl;
  return 0;
}
