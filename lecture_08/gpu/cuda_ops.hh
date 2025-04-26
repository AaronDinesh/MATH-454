#ifndef __CUDA_OPS_H_
#define __CUDA_OPS_H_

#include <cuda_runtime.h>


template <typename T>
/**
 * @brief Performs elementwise addition of two arrays. 
 * Must be spawned with a 2d grid and atleast as many threads as elements in the array (n)
 * 
 * @tparam T datatype
 * @param c The output array
 * @param a The first input array
 * @param b The second input array
 * @param n The number of elements
 */
__global__ void vector_add(T* c, const T* a, const T* b, size_t n);

template <typename T>
/**
 * @brief Performs the daxpy operation y = a * x + y. 
 * Must be spawned with a 2d grid and atleast as many threads as elements in the array (n)
 * x and y must be allocated on the device
 * 
 * @tparam T datatype
 * @param alpha The scalar value
 * @param x The first input array
 * @param y The output array (is also used in the operation)
 * @param n The number of elements
 */
__global__ void cu_daxpy(const T alpha, const T* x, T* y, size_t n);

template <typename T>
/**
 * @brief Performs the daxpy operation y = alpha * x + y. 
 * Must be spawned with a 2d grid and atleast as many threads as elements in the array (n)
 * alpha, x and y must be allocated on the device
 * 
 * @tparam T datatype
 * @param alpha The scalar value
 * @param x The first input array
 * @param y The output array (is also used in the operation)
 * @param n The number of elements
 */

__global__ void cu_daxpy(const T* alpha, const T* x, T* y, size_t n);

template <typename T>
/**
 * @brief Performs the ddot operation c = x * y. 
 * Must be spawned with a 2d grid and atleast as many threads as elements in the array (n)
 * Remember to ZERO the c array before calling this function.
 * 
 * @param c The output array
 * @param x The first input array
 * @param y The second input array
 * @param n The number of elements
 */
__global__ void cu_ddot(T* c, const T* x, const T* y, size_t n);

template <typename T>
/**
 * @brief Performs the dgemv operation c = alpha * A * x + beta * c. 
 * Must be spawned with a 2d grid and atleast as many threads as elements in the array (n)
 * 
 * @param c The output array
 * @param A The input matrix
 * @param x The input array
 * @param alpha The scalar value for the A*x product
 * @param beta The scalar value for c
 * @param m The number of rows
 * @param n The number of columns
 */
__global__ void cu_dgemv(T* c, const T* A, const T* x, const T alpha, const T beta, size_t m, size_t n);

template <typename T>
/**
 * @brief Performs the dgemm operation C = alpha * A * B + beta * C. (A is m x k and B is k x n)
 * Must be spawned with a 2d grid and atleast as many threads as elements in the array (n). 
 * 
 * @param C The output matrix
 * @param A The first input matrix
 * @param B The second input matrix
 * @param alpha The scalar value for the A*B product
 * @param beta The scalar value for C
 * @param m The number of rows in A
 * @param n The number of columns in B
 * @param k The number of columns in A and rows in B
 */
__global__ void cu_dgemm(T* C, const T* A, const T* B, const T alpha, const T beta, size_t m, size_t n, size_t k);


template <typename T>
/**
 * @brief Negates the elements of a vector on device memory. If n=1 it negates a scalar
 * It is up to the user to ensure that the correct number of threads and blocks are used
 * for the mode specified in config.hh
 * 
 * @tparam T datatype
 * @param a pointer to the vector
 * @param n number of elements
 */
__global__ void cu_negate(T* a, size_t n);


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
 * @brief Computes the alpha value for the CG algorithm. Must be launched with a 2D grid and at least as many threads as
 * elements in the array (n). It should be launched as compute_alpha<<<1,1>>>
 * 
 * @tparam T datatype
 * @param alpha pointer to the alpha value
 * @param rsold pointer to the rsold value
 * @param d_pAp pointer to the d_pAp value
 */
__global__ void compute_alpha(T* alpha, const T* rsold, const T* d_pAp);

template <typename T>
/**
 * @brief Computes the beta value for the CG algorithm. Must be launched with a 2D grid and at least as many threads as
 * elements in the array (n). It should be launched as compute_beta<<<1,1>>>
 * 
 * @tparam T datatype
 * @param beta pointer to the beta value
 * @param rsold pointer to the rsold value
 * @param rsnew pointer to the rsnew value
 */
__global__ void compute_beta(T* beta, const T* rsold, const T* rsnew);

template <typename T>
/**
 * @brief Frees a block of allocated memory on the CUDA device
 * 
 * @tparam T datatype
 * @param d_a pointer to the device array
 */
__host__ void free_on_device(T* &d_a);

#endif