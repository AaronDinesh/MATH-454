NVCC       = nvcc
CXX       ?= g++
CXXFLAGS  += -O3 -Wall
NVCCFLAGS += -O3
LDFLAGS   += $(NVCCFLAGS)

#Target Rules
hello: hello_world.o
	$(NVCC) $^ $(LDFLAGS) -o $@

%.o:%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf *.o hello_world
