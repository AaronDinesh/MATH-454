#include <iostream>
#include <mpi.h>

int main(int argc, char **argv) {
    int rank, size;
    
    //Initalize the MPI env
    int ret_code = MPI_Init(&argc, &argv);

    if(ret_code == MPI_SUCCESS){
        std::cout << "MPI initialized successfully" << std::endl;
    }else{
        std::cerr << "MPI initialization failed" << std::endl;
        return -1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::cout << "Hello world from process " << rank << " of " << size << std::endl;
    MPI_Finalize();
    return 0;
}