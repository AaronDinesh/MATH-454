#pragma once

#include <cuda_runtime.h>

/**
* @brief Computes the time step size that satisfied the CFL condition on the GPU
* 
* @param dptr_h Water height in the current time step
* @param dptr_hu x water velocity in the current time step
* @param dptr_hv y water velocity in the current time step
* @param nx Number of cells along the x direction
* @param ny Number of cells along the y direction
* @param g Gravity constant
* @param dptr_partial_max Pointer to the partial max value array
*/

__global__ void compute_time_step_gpu(const double* __restrict__ dptr_h, const double* __restrict__ dptr_hu, const double* __restrict__ dptr_hv, int nx, int ny, double g, double* dptr_partial_max);


/**
* @brief Solve one step of the SWE.
* @note This function is a parallelized version of the solve_step function.
* 
* @param dptr_h0 Water height in the previous time step
* @param dptr_hu0 x water velocity in the previous time step
* @param dptr_hv0 y water velocity in the previous time step
* @param dptr_zdx zdx
* @param dptr_zdy zdy
* @param dptr_h Water height in the current time step
* @param dptr_hu x water velocity in the current time step
* @param dptr_hv y water velocity in the current time step
* @param nx Number of cells along the x direction
* @param ny Number of cells along the y direction
* @param dt Time step
* @param size_x Size in km along the x direction
* @param size_y Size in km along the y direction
* @param g Gravity constant
*/
__global__ void solve_step_mega_kernel(const double * __restrict__ dptr_h0, const double * __restrict__ dptr_hu0, const double * __restrict__ dptr_hv0, const double * __restrict__ dptr_zdx, const double * __restrict__ dptr_zdy, double * __restrict__ dptr_h, double * __restrict__ dptr_hu, double * __restrict__ dptr_hv, int nx, int ny, double dt, double size_x, double size_y, double g);

/**
* @brief Update boundary conditions on the GPU along the vertical axis
* 
* @param dptr_h0_ Water height in the previous time step
* @param dptr_hu0_ x water velocity in the previous time step
* @param dptr_hv0_ y water velocity in the previous time step
* @param dptr_h_ Water height in the current time step
* @param dptr_hu_ x water velocity in the current time step
* @param dptr_hv_ y water velocity in the current time step
* @param nx_ Number of cells along the x direction
* @param ny_ Number of cells along the y direction
* @param coef Coefficient
*/
__global__ void update_bcs_vert_gpu(const double* dptr_h0_, const double* dptr_hu0_, const double* dptr_hv0_, double* dptr_h_, double* dptr_hu_, double* dptr_hv_, int nx_, int ny_, double coef);


/**
* @brief Update boundary conditions on the GPU along the horizontal axis
* 
* @param dptr_h0_ Water height in the previous time step
* @param dptr_hu0_ x water velocity in the previous time step
* @param dptr_hv0_ y water velocity in the previous time step
* @param dptr_h_ Water height in the current time step
* @param dptr_hu_ x water velocity in the current time step
* @param dptr_hv_ y water velocity in the current time step
* @param nx_ Number of cells along the x direction
* @param ny_ Number of cells along the y direction
* @param coef Coefficient 
*/
__global__ void update_bcs_horiz_gpu(const double* dptr_h0_, const double* dptr_hu0_, const double* dptr_hv0_, double* dptr_h_, double* dptr_hu_, double* dptr_hv_, int nx_, int ny_, double coef);


template <typename T>
  /**
  * @brief Copies data from the CUDA device to the host
  * 
  * @tparam T datatype
  * @param h_a pointer to the host array
  * @param d_a pointer to the device array
  * @param count number of elements to copy
  */
  __host__ void copy_from_device(T* &h_a, const T* d_a, size_t count);


  template <typename T>
  /**
  * @brief Copies data from the host to the CUDA device
  * 
  * @tparam T datatype
  * @param d_a pointer to the device array
  * @param h_a pointer to the host array
  * @param count number of elements to copy
  */
  __host__ void copy_to_device(T* &d_a, const T* h_a, size_t count);


  template <typename T>
  /**
  * @brief Creates a zeroed array on the CUDA device
  * 
  * @tparam T datatype
  * @param d_a pointer to the device array
  * @param count number of elements
  */
  __host__ void zero_malloc_on_device(T* &d_a, size_t count);


  template <typename T>
  /**
  * @brief Copies data from one device array to another
  * 
  * @tparam T datatype
  * @param d_a pointer to the source device array
  * @param d_b pointer to the destination device array
  * @param count number of elements
  */
  __host__ void assign_on_device(const T* src, T* dst, size_t count);


  template <typename T>
  /**
  * @brief Assigns a scalar value to a device variable
  * 
  * @tparam T datatype
  * @param a scalar value
  * @param dst pointer to the device variable
  */
  __host__ void assign_on_device(const T a, T* dst);

  template <typename T>
  /** 
  * @brief Assigns a scalar value to all the elements of a array
  * 
  * @tparam T datatype
  * @param a scalar value
  * @param dst pointer to the device array
  * @param count number of elements
  */
  __host__ void assign_on_device(const T a, T* dst, size_t count);

  template <typename T>
  /**
  * @brief Fills and array with zeros.
  * 
  * @tparam T datatype
  * @param dst pointer to the array
  * @param count number of elements
  */

  __host__ void zero_fill_array(T* dst, size_t count);

  template <typename T>
  /**
  * @brief Frees a block of allocated memory on the CUDA device
  * 
  * @tparam T datatype
  * @param d_a pointer to the device array
  */
  __host__ void free_on_device(T* &d_a);

