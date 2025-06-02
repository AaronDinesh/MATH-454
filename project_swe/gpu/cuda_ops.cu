#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include "cuda_ops.hh"
#include "config.hh"
#include <iostream>

__global__ void compute_time_step_gpu(const double* __restrict__ dptr_h, const double* __restrict__ dptr_hu, const double* __restrict__ dptr_hv, int nx, int ny, double g, double* dptr_partial_max){
  extern __shared__ double shared_data[];

  // Get my thread and block ids
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bdx = blockDim.x;
  int bdy = blockDim.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Global indices, offset by +1 to skip boundaries
  int i = bx * bdx + tx + 1;  // i ∈ [1..nx−2]
  int j = by * bdy + ty + 1;  // j ∈ [1..ny−2]

  double nu2 = 0.0;
  if (i < nx-1 && j < ny-1) {
    double hij  = dptr_h[j * nx + i];
    double huij = dptr_hu[j * nx + i];
    double hvij = dptr_hv[j * nx + i];
    double sqrt_h = sqrt(hij);
    double nu_u = fabs(huij)/hij + sqrt_h;
    double nu_v = fabs(hvij)/hij + sqrt_h;
    nu2 = nu_u * nu_u + nu_v * nu_v;
  }

  int tid = ty * bdx + tx;
  shared_data[tid] = nu2;
  __syncthreads();

  int blockSize = bdx * bdy;
  for(int s = blockSize >> 1; s > 0; s >>= 1){
    if(tid < s){
      shared_data[tid] = fmax(shared_data[tid], shared_data[tid + s]);
    }
    __syncthreads();
  }

  if(tid == 0){
    int blockId = bx + by * gridDim.x;
    dptr_partial_max[blockId] = shared_data[0];
  }
}


__global__ void solve_step_mega_kernel(const double * __restrict__ dptr_h0,
                                                  const double * __restrict__ dptr_hu0,
                                                  const double * __restrict__ dptr_hv0,
                                                  const double * __restrict__ dptr_zdx,
                                                  const double * __restrict__ dptr_zdy,
                                                        double * __restrict__ dptr_h,
                                                        double * __restrict__ dptr_hu,
                                                        double * __restrict__ dptr_hv,
                                                  int nx, int ny,
                                                  double dt,
                                                  double size_x,
                                                  double size_y,
                                                  double g)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bdx = blockDim.x;
  int bdy = blockDim.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Global indices, offset by +1 to skip boundaries
  int i = bx * bdx + tx + 1;  // i in [1..nx−2]
  int j = by * bdy + ty + 1;  // j in [1..ny−2]

  if(i >= 1 && i < nx-1 && j >= 1 && j < ny-1){
    // Create some constants for the kernel
    double dx = size_x / double(nx);
    double dy = size_y / double(ny);
    double C1x = 0.5 * dt / dx;
    double C1y = 0.5 * dt / dy;
    double C2 = dt * g;
    double C3 = 0.5 * g;

    // Find out the indices of the neighbors
    int idx = j * nx + i;
    int idx_left = idx - 1;
    int idx_right = idx + 1;
    int idx_down = (j - 1) * nx + i;
    int idx_up = (j + 1) * nx + i;

    double h0_left = dptr_h0[idx_left];
    double h0_right = dptr_h0[idx_right];
    double h0_down = dptr_h0[idx_down];
    double h0_up = dptr_h0[idx_up];

    double hu0_left = dptr_hu0[idx_left];
    double hu0_right = dptr_hu0[idx_right];
    double hv0_down = dptr_hv0[idx_down];
    double hv0_up = dptr_hv0[idx_up];

    double hij = 0.25 * (h0_down + h0_up + h0_left + h0_right)
                 + C1x * (hu0_left - hu0_right)
                 + C1y * (hv0_down - hv0_up);
  
    if (hij < 0.0){
      hij = 1.0e-5;
    }

    dptr_h[idx] = hij;

    if (hij > 1e-4){
      double hu0_down = dptr_hu0[idx_down];
      double hu0_up = dptr_hu0[idx_up];
      double hv0_left = dptr_hv0[idx_left];
      double hv0_right = dptr_hv0[idx_right];
    
      double hu0_left_sq = hu0_left * hu0_left / h0_left;
      double hu0_right_sq = hu0_right * hu0_right / h0_right;
      double hv0_down_sq = hv0_down * hv0_down / h0_down;
      double hv0_up_sq = hv0_up * hv0_up / h0_up;

      double hu_flux_x = 0.25 * (hu0_down + hu0_up + hu0_left + hu0_right)
                         - C2 * hij * dptr_zdx[idx]
                         + C1x * (hu0_left_sq + C3*(dptr_h0[idx_left] * dptr_h0[idx_left])
                                 - hu0_right_sq - C3*(dptr_h0[idx_right] * dptr_h0[idx_right]))
                         + C1y * ((hu0_down * hv0_down) / dptr_h0[idx_down]
                                 - (hu0_up * hv0_up) / dptr_h0[idx_up]);

      double hv_flux_y = 0.25 * (hv0_down + hv0_up + hv0_left + hv0_right)
                        - C2 * hij * dptr_zdy[idx]
                        + C1x * ((hu0_left * hv0_left) / dptr_h0[idx_left]
                                - (hu0_right * hv0_right) / dptr_h0[idx_right])
                        + C1y * (hv0_down_sq + C3*(dptr_h0[idx_down] * dptr_h0[idx_down])
                                - hv0_up_sq - C3*(dptr_h0[idx_up] * dptr_h0[idx_up]));

      dptr_hu[idx] = hu_flux_x;
      dptr_hv[idx] = hv_flux_y;

    }else{
      dptr_hu[idx] = 0.0;
      dptr_hv[idx] = 0.0;
    }
  }
}

__global__ void update_bcs_vert_gpu(const double* dptr_h0_, const double* dptr_hu0_, const double* dptr_hv0_, double* dptr_h_, double* dptr_hu_, double* dptr_hv_, int nx_, int ny_, double coef){
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < nx_){
    // This is for the top boundary: j = 0

    int idx_below = nx_ + i;
    dptr_h_[i] = dptr_h0_[idx_below];
    dptr_hu_[i] = dptr_hu0_[idx_below];
    dptr_hv_[i] = coef * dptr_hv0_[idx_below];

    int idx_bottom = (ny_ - 1) * nx_ + i;
    int idx_just_above_bottom = (ny_ - 2) * nx_ + i;
    dptr_h_[idx_bottom] = dptr_h0_[idx_just_above_bottom];
    dptr_hu_[idx_bottom] = dptr_hu0_[idx_just_above_bottom];
    dptr_hv_[idx_bottom] = coef * dptr_hv0_[idx_just_above_bottom];
  }
}

__global__ void update_bcs_horiz_gpu(const double* dptr_h0_, const double* dptr_hu0_, const double* dptr_hv0_, double* dptr_h_, double* dptr_hu_, double* dptr_hv_, int nx_, int ny_, double coef){
 int j = blockIdx.x * blockDim.x + threadIdx.x;

  if(j < ny_){
    // This is for the left boundary: i = 0

    int idx_left = j * nx_;
    dptr_h_[idx_left] = dptr_h0_[idx_left + 1];
    dptr_hu_[idx_left] = coef * dptr_hu0_[idx_left + 1];
    dptr_hv_[idx_left] = dptr_hv0_[idx_left + 1];

    int idx_right = j * nx_ + (nx_ - 1);
    int idx_just_left_of_right = j * nx_ + (nx_ - 2);
    dptr_h_[idx_right] = dptr_h0_[idx_just_left_of_right];
    dptr_hu_[idx_right] = coef * dptr_hu0_[idx_just_left_of_right];
    dptr_hv_[idx_right] = dptr_hv0_[idx_just_left_of_right];
 } 
}


template <typename T>
__host__ void copy_from_device(T* &h_a, const T* d_a, size_t count){
  size_t size = count * sizeof(T);

  if (!h_a) {
    h_a = (T*) malloc(size);
    //cudaMallocHost((void **)& h_a, size);
  }

  #if ERROR_CHECKING
    if (!h_a) {
        std::cout << "Host malloc failed\n";
        return;
    }
  #endif

  cudaError_t cudaStatus = cudaMemcpy(static_cast<void *>(h_a), static_cast<const void*>(d_a), size, cudaMemcpyDeviceToHost); 
  
  #if ERROR_CHECKING
    if (cudaStatus != cudaSuccess) {
      free(h_a);
      h_a = nullptr;
      std::cout << "copy_from_device failed because cudaMemcpy failed! Cuda error below." << std::endl;
      std::cout << "Cuda error:" << cudaGetErrorString(cudaStatus) << std::endl;
    }
  #endif
}

template <typename T>
__host__ void copy_to_device(T* &d_a, const T* h_a, size_t count){
  size_t size = count * sizeof(T);

  cudaError_t cudaStatus;

  if(!d_a){ 
    cudaStatus = cudaMalloc((void**) &d_a, size);
  
    #if ERROR_CHECKING
      if (cudaStatus != cudaSuccess) {
        std::cout << "copy_to_device failed because cudaMalloc failed! Cuda error below." << std::endl;
        std::cout << "Cuda error:" << cudaGetErrorString(cudaStatus) << std::endl;
      }
    #endif
  }

  cudaStatus = cudaMemcpy((void*) d_a, (const void*) h_a, size, cudaMemcpyHostToDevice);
  
  #if ERROR_CHECKING
    if (cudaStatus != cudaSuccess) {
      cudaFree((void*) d_a);
      d_a = nullptr;
      std::cout << "copy_to_device failed because cudaMemcpy failed! Cuda error below." << std::endl;
      std::cout << "Cuda error:" << cudaGetErrorString(cudaStatus) << std::endl;
    }
  #endif
}

template <typename T>
__host__ void zero_malloc_on_device(T* &d_a, size_t count){
  size_t size = count * sizeof(T);

  cudaError_t cudaStatus = cudaMalloc((void**) &d_a, size);

  #if ERROR_CHECKING
    if (cudaStatus != cudaSuccess){
      std::cout << "zero_malloc_on_device failed because cudaMalloc failed! Cuda error below." << std::endl;
      std::cout << "Cuda error:" << cudaGetErrorString(cudaStatus) << std::endl;
    }
  #endif

  cudaStatus = cudaMemset((void*) d_a, 0, size);

  #if ERROR_CHECKING
    if(cudaStatus != cudaSuccess){
      cudaFree((void*) d_a);
      d_a = nullptr;
      std::cout << "zero_malloc_on_device failed because cudaMemset failed! Cuda error below." << std::endl;
      std::cout << "Cuda error:" << cudaGetErrorString(cudaStatus) << std::endl; 
    }
  #endif
}


template <typename T>
__host__ void assign_on_device(const T* src, T* dst, size_t count){
  size_t size = count * sizeof(T);

  cudaError_t cudaStatus = cudaMemcpy((void*) dst, (const void*) src, size, cudaMemcpyDeviceToDevice);
  
  #if ERROR_CHECKING
    if (cudaStatus != cudaSuccess) {
      std::cout << "assign_on_device failed! Cuda error below." << std::endl;
      std::cout << "Cuda error:" << cudaGetErrorString(cudaStatus) << std::endl;
    }
  #endif
}

template <typename T>
__host__ void assign_on_device(const T a, T* dst){
  cudaError_t cudaStatus = cudaMemcpy((void*) dst, (const void*) &a, sizeof(T), cudaMemcpyHostToDevice);
  
  #if ERROR_CHECKING
    if (cudaStatus != cudaSuccess) {
      std::cout << "assign_on_device failed! Cuda error below." << std::endl;
      std::cout << "Cuda error:" << cudaGetErrorString(cudaStatus) << std::endl;
    }
  #endif
}

template <typename T>
__host__  void zero_fill_array(T* dst, size_t count){
  size_t size = count * sizeof(T);

  cudaError_t cudaStatus = cudaMemset((void*) dst, (T) 0, size);

  #if ERROR_CHECKING
    if (cudaStatus != cudaSuccess) {
      std::cout << "zero_fill_array failed! Cuda error below." << std::endl;
      std::cout << "Cuda error:" << cudaGetErrorString(cudaStatus) << std::endl;
    }
  #endif
}

template <typename T>
__host__ void free_on_device(T* &d_a){
  if(d_a){
    cudaError_t cudaStatus = cudaFree((void*) d_a);
    d_a = nullptr;
    #if ERROR_CHECKING
    if (cudaStatus != cudaSuccess) {
      std::cout << "free_on_device failed! Cuda error below." << std::endl;
      std::cout << "Cuda error:" << cudaGetErrorString(cudaStatus) << std::endl;
    }
    #endif
  }
}


// For device functions
template void copy_to_device<double>(double*&, const double*, size_t);
template void copy_to_device<bool>(bool*&, const bool*, size_t);
template void copy_to_device<std::size_t>(std::size_t*&, const std::size_t*, size_t);
//template void copy_from_device<double>(double*, const double*, size_t);
template void copy_from_device<double>(double*&, const double*, size_t);
template void zero_malloc_on_device<double>(double*&, size_t);
template void assign_on_device<double>(const double*, double*, size_t);
template void assign_on_device<double>(const double, double*);
template void free_on_device<double>(double*&);
template void zero_fill_array<double>(double*, size_t);
