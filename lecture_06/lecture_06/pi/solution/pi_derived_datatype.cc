/*
  This exercise is taken from the class Parallel Programming Workshop (MPI,
  OpenMP and Advanced Topics) at HLRS given by Rolf Rabenseifner
 */

#include <chrono>
#include <cstdio>
#include <cmath>
#include <mpi.h>
#include <vector>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

inline int digit(double x, int n) {
  return std::trunc(x * std::pow(10., n)) - std::trunc(x * std::pow(10., n - 1)) *10.;
}

inline double f(double a) { return (4. / (1. + a * a)); }

const int n = 10000000;

int main(int /* argc */ , char ** /* argv */) {
  int i;
  double dx, x, l_sum, pi;
  int psize, prank;

  MPI_Init(NULL, NULL);

  MPI_Comm_size(MPI_COMM_WORLD, &psize);
  MPI_Comm_rank(MPI_COMM_WORLD, &prank);

  struct Sum {
    double sum;
    int rank;
  };

  Sum sum{0., 0};

  int blk_length[2] = {1, 1};

  MPI_Aint zero_address, first_address, second_address;
  MPI_Get_address(&sum, &zero_address);
  MPI_Get_address(&sum.sum, &first_address);
  MPI_Get_address(&sum.rank, &second_address);

  MPI_Aint displs[2];
  displs[0] = MPI_Aint_diff(first_address, zero_address);;
  displs[1] = MPI_Aint_diff(second_address, first_address);

  MPI_Datatype types[2] = {MPI_DOUBLE, MPI_INT};
  MPI_Datatype sum_t;
  MPI_Type_create_struct(2, blk_length, displs, types, &sum_t);
  MPI_Type_commit(&sum_t);

  auto mpi_t1 = MPI_Wtime();
  auto t1 = clk::now();

  int nlocal = n / psize;
  int istart = 1 + nlocal * prank;
  int iend = nlocal * (prank + 1);

  /* calculate pi = integral [0..1] 4 / (1 + x**2) dx */
  dx = 1. / n;
  l_sum = 0.0;
  for (i = istart; i <= iend; i++) {
    x = (1. * i - 0.5) * dx;
    l_sum = l_sum + f(x);
  }

  sum.sum = l_sum;
  sum.rank = prank;

  if(prank == 0) {
    std::vector<Sum> sums(psize);
    MPI_Gather(&sum, 1, sum_t, sums.data(), 1, sum_t, 0, MPI_COMM_WORLD);

    l_sum = 0.;
    for(auto s : sums) {
      l_sum += s.sum;
    }

    sum.sum = l_sum;
  } else {
    MPI_Gather(&sum, 1, sum_t, NULL, 0, sum_t, 0, MPI_COMM_WORLD);
  }


  MPI_Bcast(&sum, 1, sum_t, 0, MPI_COMM_WORLD);

  pi = dx * sum.sum;

  auto mpi_elapsed = MPI_Wtime() - mpi_t1;
  second elapsed = clk::now() - t1;

  if(prank == 0) {
    std::printf("computed pi                 = %.16g\n", pi);
    std::printf("wall clock time (mpi_wtime) = %.4gs with %d process\n", mpi_elapsed, psize);
    std::printf("wall clock time (chrono)    = %.4gs\n", elapsed.count());

    for(int d = 1; d <= 15; ++d) {
      std::printf("%d", digit(pi, d));
    }
  }

  MPI_Type_free(&sum_t);
  MPI_Finalize();

  return 0;
}
