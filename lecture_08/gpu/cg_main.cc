#include "cg.hh"
#include "config.hh"
#include <chrono>
#include <iostream>
#include <string>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

/*
Implementation of a simple CG solver using matrix in the mtx format (Matrix
market) Any matrix in that format can be used to test the code
*/
int main(int argc, char ** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " [martix-market-filename] [threads-per-block]"
              << std::endl;
    return 1;
  }

  CGSolver solver;
  solver.read_matrix(argv[1]);

  int n = solver.n();
  int m = solver.m();
  double h = 1. / n;

  solver.init_source_term(h);

  std::vector<double> x_d(n);
  std::fill(x_d.begin(), x_d.end(), 0.);

  #if SCALING_EXPERIMENTS == 0
    std::cout << "Call CG dense on matrix size " << m << " x " << n << ")"
                << std::endl;
  #endif

  auto t1 = clk::now();
  solver.solve(x_d, std::stoi(argv[2]));
  second elapsed = clk::now() - t1;
  
  #if SCALING_EXPERIMENTS == 0
    std::cout << "Time for CG (dense solver)  = " << elapsed.count() << " [s]\n";
  #else
    std::cout << std::stoi(argv[2]) << "," << elapsed.count() << std::endl;
  #endif
  
  return 0;
}
