// includes, system
#include <chrono>
#include <iostream>

#include "matrix_mul.hh"

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

/* -------------------------------------------------------------------------- */
int main(int argc, char **argv) {

  // By default, we use device 0,
  int dev_id = 0;

  cudaError error;
  cudaDeviceProp device_prop;
  error = cudaGetDevice(&dev_id);
  error = cudaGetDeviceProperties(&device_prop, dev_id);
  if (device_prop.computeMode == cudaComputeModeProhibited) {
    std::cerr << "Error: device is running in <Compute Mode Prohibited>, no "
                 "threads can use ::cudaSetDevice()"
              << std::endl;
    exit(EXIT_SUCCESS);
  }

  if (error != cudaSuccess) {
    std::cout << "cudaGetDeviceProperties returned error code " << error
              << ", line(" << __LINE__ << ")" << std::endl;
  } else {
    std::cout << "GPU Device " << dev_id << ": \"" << device_prop.name
              << "\" with compute capability " << device_prop.major << "."
              << device_prop.minor << std::endl;
  }

    // allocate device memory
  matrix_gpu device_A(HA, WA);
  matrix_gpu device_B(HB, WB);

  std::mt19937 gen(2006);
  std::uniform_real_distribution<> dis(0.f, 1.f);

  float flop = 2.f * WC * HC * WA;

  // initialize host memory
  device_A.randomInit(gen, dis);
  device_B.randomInit(gen, dis);

#if CHECK_RESULT == 1
  matrix_cpu host_C(HC, WC);

  auto t1 = clk::now();
  // compute reference solution
  matMulCPU(host_C, device_A, device_B);

  second elapsed = clk::now() - t1;

  std::cout << "Naive CPU -- time:  " << elapsed.count()
            << " (s), GFLOPs: " << flop / elapsed.count() / 1e9 << std::endl;
#endif

  /****************************************************/
  /*  naive implementation on GPU                     */
  /****************************************************/
  matrix_gpu device_C(HC, WC);

  // setup execution parameters
  dim3 threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid = dim3(device_C.cols() / threads.x, device_C.rows() / threads.y);

  cudaEvent_t start;
  cudaEvent_t stop;
  float msec_total;

  // create and start timer
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  // naive implementation
  matMulGPU_naive<<<grid, threads>>>(device_C, device_A, device_B);

  // stop and destroy timer
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msec_total, start, stop);

  cudaDeviceSynchronize();
  std::cout << "Naive GPU -- time:  " << msec_total / 1e3
            << " (s), GFLOPs: " << flop / msec_total / 1e6 << std::endl;

#if CHECK_RESULT == 1
  // check result
  printDiff(host_C, device_C);
#endif

  return 0;
}
