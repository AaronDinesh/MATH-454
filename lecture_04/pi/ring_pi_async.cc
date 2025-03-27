/*
  This exercise is taken from the class Parallel Programming Workshop (MPI,
  OpenMP and Advanced Topics) at HLRS given by Rolf Rabenseifner
 */

#include <chrono>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <mpi.h>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

inline int digit(double x, int n) {
  return std::trunc(x * std::pow(10., n)) - std::trunc(x * std::pow(10., n - 1)) *10.;
}

inline double f(double a) { return (4. / (1. + a * a)); }

const int n = 10000000;

int main(int argc, char** argv) {
  int rank, size;
  
  //Initalize the MPI env
  int ret_code = MPI_Init(&argc, &argv);

  if(ret_code != MPI_SUCCESS){
    std::cerr << "MPI initialization failed" << std::endl;
    return -1;
  }

  //Get rank and size
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int i;
  double dx, x, local_sum, pi;

  auto t1 = clk::now();

  //Divide all the parts and let the last node pick up the remainder
  int my_part;
  int step_size = (int) n / size;
  my_part = (rank == size-1)  ? step_size + n % size : step_size;

  /* calculate pi = integral [0..1] 4 / (1 + x**2) dx */
  dx = 1. / n;
  local_sum = 0.0;
  for (i = rank*step_size; i < rank*step_size + my_part; i++) {
    x = 1. * i * dx;
    local_sum = local_sum + f(x);
  }
  pi = dx * local_sum;

  //Now we need to send our local sum to each other in a ring clockwise
  // We need to make size-1 comms
  
  double recv_buff = 0;
  MPI_Request request;
  for(int i=1; i < size; i++){
    int rank_to_send_to = (rank + i) % size;
    int rank_to_recv_from = (rank - i + size) % size;
    //MPI_Sendrecv(&local_sum, 1, MPI_DOUBLE, rank_to_send_to, 0, &recv_buff, 1, MPI_DOUBLE, rank_to_recv_from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Isend(&local_sum, 1, MPI_DOUBLE, rank_to_send_to, 0, MPI_COMM_WORLD, &request);
    MPI_Recv(&recv_buff, 1, MPI_DOUBLE, rank_to_recv_from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    pi += recv_buff*dx;
  }

  if(rank==0){
    second elapsed = clk::now() - t1;

    std::printf("computed pi                     = %.16g\n", pi);
    std::printf("wall clock time (chrono)        = %.4gs\n", elapsed.count());

    for(int d = 1; d <= 15; ++d) {
      std::printf("%d", digit(pi, d));
    }
    std::printf("\n");
  }

  MPI_Finalize();
  return 0;
}
