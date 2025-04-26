# Compiler and flags
CXX = g++
NVCC = nvcc
CC = gcc

# Output executable name
TARGET = cg_solver

# Compiler flags as specified
CXXFLAGS := -std=c++11 -O3 -Wall -Wextra -Werror -Wno-unused-function -Wno-unused-variable -pedantic 
CCFLAGS := -O3 
NVCCFLAGS := -std=c++11 -O3 --Werror cross-execution-space-call,deprecated-declarations \
             --gpu-architecture=compute_70 --gpu-code=sm_70 --use_fast_math

# CUDA specific link flags
CUDA_LDFLAGS := -lcudart

# Include directories
INCLUDES = -I.

# Source files by type
CXX_SOURCES = cg_main.cc matrix.cc matrix_coo.cc
NVCC_SOURCES = cg.cu cuda_ops.cu
C_SOURCES = mmio.c

# Object files
CXX_OBJECTS = $(CXX_SOURCES:.cc=.o)
NVCC_OBJECTS = $(NVCC_SOURCES:.cu=.o)
C_OBJECTS = $(C_SOURCES:.c=.o)

# All object files
OBJECTS = $(CXX_OBJECTS) $(NVCC_OBJECTS) $(C_OBJECTS)

# Default target
all: $(TARGET)

# Linking rule - important: cuda_ops.o must come before cg.o to resolve template dependencies
$(TARGET): $(C_OBJECTS) $(CXX_OBJECTS) cuda_ops.o cg.o
	$(NVCC) $(NVCCFLAGS) $(CUDA_LDFLAGS) -o $@ cuda_ops.o cg.o $(C_OBJECTS) $(CXX_OBJECTS)

# Compilation rules for C++ files
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compilation rules for CUDA files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Compilation rules for C files
%.o: %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

# Special rule for cuda_ops.cu to ensure template instantiation
cuda_ops.o: cuda_ops.cu cuda_ops.hh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) --device-c -c $< -o $@

# Special rule for cg.cu to ensure it links with cuda_ops.o
cg.o: cg.cu cg.hh matrix.hh cuda_ops.hh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJECTS) $(TARGET)

# Dependencies
cg_main.o: cg_main.cc cg.hh matrix.hh matrix_coo.hh config.hh
matrix.o: matrix.cc matrix.hh
matrix_coo.o: matrix_coo.cc matrix_coo.hh matrix.hh mmio.h
mmio.o: mmio.c mmio.h

.PHONY: all clean
