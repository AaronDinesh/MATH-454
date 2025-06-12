#include "swe.hh"
#include <string>
#include <cstddef>
#include <iostream>
#include <chrono>
#include <mpi.h>

int main(int argc, char* argv[]){
  // Option 1 - Solving simple problem: water drops in a box
  const int test_case_id = std::stoi(argv[1]);
  const double Tend = 1.0;
  const std::size_t nx = std::stoi(argv[2]);
  const std::size_t ny = std::stoi(argv[3]);
  const std::size_t output_n = std::stoi(argv[4]);
  const bool full_log = false;

  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if(test_case_id == 1){
    const std::string output_fname = "./output_files/water_drops";
    std::chrono::_V2::system_clock::time_point start;
    SWESolver solver(test_case_id, nx, ny, MPI_COMM_WORLD);
    if (rank == 0){
      start = std::chrono::high_resolution_clock::now();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    solver.solve(Tend, full_log, output_n, output_fname);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){  
      auto end = std::chrono::high_resolution_clock::now();
    
      auto duration = std::chrono::duration<double>(end - start);
      std::cout << "Time taken: " << duration.count() << " s" << std::endl;
    }
  }else if(test_case_id == 2){
    // Option 2 - Solving analytical (dummy) tsunami example.
    const std::string output_fname = "./output_files/analytical_tsunami";
    std::chrono::_V2::system_clock::time_point start;
    SWESolver solver(test_case_id, nx, ny, MPI_COMM_WORLD);
    if (rank == 0){
      start = std::chrono::high_resolution_clock::now();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    solver.solve(Tend, full_log, output_n, output_fname);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){  
      auto end = std::chrono::high_resolution_clock::now();
    
      auto duration = std::chrono::duration<double>(end - start);
      std::cout << "Time taken: " << duration.count() << " s" << std::endl;
    }    
  }else if(test_case_id == 3){
  // Option 3 - Solving tsunami problem with data loaded from file.
  const double Tend = 0.2;   // Simulation time in hours
  const double size = 500.0; // Size of the domain in km

  // const std::string fname = "Data_nx501_500km.h5"; // File containg initial data (501x501 mesh).
  const std::string fname = "Data_nx1001_500km.h5"; // File containg initial data (1001x1001 mesh).
  // const std::string fname = "Data_nx2001_500km.h5"; // File containg initial data (2001x2001 mesh).
  // const std::string fname = "Data_nx4001_500km.h5"; // File containg initial data (4001x4001 mesh).
  // const std::string fname = "Data_nx8001_500km.h5"; // File containg initial data (8001x8001 mesh).

  const std::string output_fname = "./output_files/tsunami";
  const bool full_log = false;

  SWESolver solver(fname, size, size, MPI_COMM_WORLD);
  solver.solve(Tend, full_log, output_n, output_fname);
  }else{
    std::cout << "Invalid Test Case Id !!" << std::endl;
  }
  MPI_Finalize();
  return 0;
}
