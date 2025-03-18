OPTIM+=-O3 -march=native

# No MPI with gcc. You have to load the module gcc
CXX=g++
CC=g++

# MPI with gcc. You have to load the modules gcc and openmpi
# CXX=mpicxx
# CC=mpicxx


# # No MPI with intel. You have to load the module intel
# CXX=icpc
# CC=icc
# This line below is required by intel compiler in the server
# CXXFLAGS+=-no-multibyte-chars

# # MPI with intel. You have to load the modules intel and intel-oneapi-mpi
# CXX=mpiicpc
# CC=mpiicc
# # This line below is required by intel compiler in the server
# CXXFLAGS+=-no-multibyte-chars



LD=${CXX}
CXXFLAGS+=-Wall -Wextra -std=c++11 $(OPTIM)
# LDFLAGS+=-lm
EXE=pi

all: clean $(EXE)

clean:
	rm -f $(EXE) *.o *~
