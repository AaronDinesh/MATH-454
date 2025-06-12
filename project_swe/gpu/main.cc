#include "swe.hh"
#include <string>
#include <cstddef>
#include <iostream>
#include <chrono>

int main(int argc, char* argv[]){
  // Option 1 - Solving simple problem: water drops in a box
  const int test_case_id = std::stoi(argv[1]);
  
  if(test_case_id == 1){
    const double Tend = 1.0;
    const std::size_t nx = std::stoi(argv[2]);
    const std::size_t ny = std::stoi(argv[3]);
    const std::size_t output_n = std::stoi(argv[4]);
    const std::string output_fname = "./output_files/water_drops";
    const bool full_log = false;


    std::chrono::_V2::system_clock::time_point start;
    SWESolver solver(test_case_id, nx, ny);
    solver.setThreadsPerBlock(std::stoi(argv[5]), std::stoi(argv[6]));
    start = std::chrono::high_resolution_clock::now();

    solver.solve(Tend, full_log, output_n, output_fname);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double>(end - start);
    std::cout << "Time taken: " << duration.count() << " s" << std::endl;
  }else if (test_case_id == 2){ 
    // Option 2 - Solving analytical (dummy) tsunami example.
    const int test_case_id = 2;  // Analytical tsunami test case
    const double Tend = 1.0;     // Simulation time in hours
    const std::size_t nx = 1000; // Number of cells per direction.
    const std::size_t ny = 1000; // Number of cells per direction.
    const std::size_t output_n = 10;
    const std::string output_fname = "./output_files/analytical_tsunami";
    const bool full_log = false;


    std::chrono::_V2::system_clock::time_point start;
    SWESolver solver(test_case_id, nx, ny);
    solver.setThreadsPerBlock(std::stoi(argv[5]), std::stoi(argv[6]));
    start = std::chrono::high_resolution_clock::now();
    solver.solve(Tend, full_log, output_n, output_fname);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start);
    std::cout << "Time taken: " << duration.count() << " s" << std::endl;
  }else if (test_case_id == 3){
    
    // Option 3 - Solving tsunami problem with data loaded from file.
    const double Tend = 0.2;   // Simulation time in hours
    const double size = 500.0; // Size of the domain in km
    
    // const std::string fname = "Data_nx501_500km.h5"; // File containg initial data (501x501 mesh).
    const std::string fname = "Data_nx1001_500km.h5"; // File containg initial data (1001x1001 mesh).
    // const std::string fname = "Data_nx2001_500km.h5"; // File containg initial data (2001x2001 mesh).
    // const std::string fname = "Data_nx4001_500km.h5"; // File containg initial data (4001x4001 mesh).
    // const std::string fname = "Data_nx8001_500km.h5"; // File containg initial data (8001x8001 mesh).
    
    const std::size_t output_n = 0;
    const std::string output_fname = "./output_files/tsunami";
    const bool full_log = false;
    
    SWESolver solver(fname, size, size);
    solver.setThreadsPerBlock(std::stoi(argv[5]), std::stoi(argv[6]));
    std::chrono::_V2::system_clock::time_point start;
    start = std::chrono::high_resolution_clock::now();
    solver.solve(Tend, full_log, output_n, output_fname);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start);
    std::cout << "Time taken: " << duration.count() << " s" << std::endl;
  }else{
    std::cerr << "Invalid test case id: " << test_case_id << std::endl;
  }
  return 0;
}
