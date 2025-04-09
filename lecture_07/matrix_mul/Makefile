NVCC       = nvcc
# CXX     = g++
CXX       = $(NVCC)
NVCCFLAGS += -O3
CXXFLAGS  += -O3 --compiler-options -Wall
LDFLAGS   += $(NVCCFLAGS)

#Target Rules
matrixmul: matrix_mul.o matrix_mul_cpu.o matrix_mul_gpu.o
	$(NVCC) $^ $(LDFLAGS) -o $@

%.o:%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf *.o matrixmul
