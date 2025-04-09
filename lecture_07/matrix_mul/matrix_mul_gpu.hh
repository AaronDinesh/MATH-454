#ifndef MATRIX_MUL_GPU_HH
#define MATRIX_MUL_GPU_HH

class matrix_gpu {
public:
  __host__ matrix_gpu(int n, int m) : n(n), m(m) {
    cudaMallocManaged(&data, m * n * sizeof(float));
  }
  __host__ __device__ matrix_gpu(matrix_gpu &M)
      : n(M.n), m(M.m), data(M.data), copy(true) {}
  __host__ ~matrix_gpu() {
    if (not copy) {
      cudaFree(data);
    }
  }
  __host__ __device__ inline float &operator()(int i, int j) {
    return data[i * n + j];
  }
  __host__ __device__ inline const float &operator()(int i, int j) const {
    return data[i * n + j];
  }

  __host__ matrix_gpu &operator=(const matrix_cpu &M) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        this->operator()(i, j) = M(i, j);
      }
    }
    return *this;
  }

  template <class random_gen, class random_dis>
  __host__ void randomInit(random_gen &gen, random_dis &dis) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        this->operator()(i, j) = dis(gen);
      }
    }
  }

  __host__ __device__ int rows() const { return m; }
  __host__ __device__ int cols() const { return n; }

private:
  int n;
  int m;
  float *data;
  bool copy{false};
};

__global__ void matMulGPU_naive(matrix_gpu C, const matrix_gpu A,
                                const matrix_gpu B);

#endif /* MATRIX_MUL_GPU_HH */
