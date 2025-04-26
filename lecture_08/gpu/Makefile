# Tools
NVCC        := nvcc
CXX         := g++
CC          := gcc

# Compiler flags
CXXFLAGS := -std=c++11 -O3 -Wall -Wextra -Werror -Wno-unused-function -Wno-unused-variable -pedantic 
CCFLAGS :=-O3 
NVCCFLAGS :=-std=c++11 -O3 --Werror cross-execution-space-call,deprecated-declarations, \
			--gpu-architecture=compute_70 --gpu-code=sm_70 --use_fast_math

# Linker flags
LDFLAGS     := -lcudart

# Include and source paths
INCLUDES    := -I. 
SRCS_CPP    := matrix_coo.cc matrix.cc cg.cc cg_main.cc mmio.cc
SRCS_CU     := cuda_ops.cu
OBJS_CPP    := $(SRCS_CPP:.cc=.o)
OBJS_CU     := $(SRCS_CU:.cu=.o)

TARGET      := cg_solver

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS_CPP) $(OBJS_CU)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@


clean:
	rm -f $(OBJS_CPP) $(OBJS_CU) $(TARGET)

