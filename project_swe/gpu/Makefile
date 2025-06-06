# Makefile for building the SWE GPU solver with nvcc (including XDMFWriter)
# -------------------------------------------------------------------------
#
#  - Compiles:  main.cc, swe.cu, xdmf_writer.cc
#  - Links against: HDF5, CUDA runtime (cudart)
#
# Adjust the HDF5 paths (INCLUDE and LIB) below to match your system.
HDF5_INC    := ${HDF5_ROOT}/include
HDF5_LIB    := ${HDF5_ROOT}/lib

NVCC        := nvcc
CXXFLAGS    := -std=c++14 -O3

INCLUDES    := -I$(HDF5_INC)
LDFLAGS     := -L$(HDF5_LIB)
LIBS        := -lhdf5 -lcudart

# Source files
SRC_CC      := main.cc xdmf_writer.cc
SRC_CU      := cuda_ops.cu swe.cu
TARGET      := swe_gpu

.PHONY: all clean

# Default target: build everything
all: $(TARGET)

# Rule: link all .cc and .cu sources into the final executable
# Note: nvcc will compile .cc files with its host C++ compiler and .cu with CUDA.
$(TARGET): $(SRC_CC) $(SRC_CU) swe.hh xdmf_writer.hh cuda_ops.hh
	@echo "==> Building $(TARGET) with nvcc"
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $(SRC_CC) $(SRC_CU) $(LIBS)

# Clean up:
clean:
	@echo "==> Cleaning up"
	rm -f $(TARGET) *.o
	find ./output_files -type f -exec rm {} +
	rm -f water_animation_3d.mp4
