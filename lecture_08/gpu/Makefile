CXX=g++
LD=${CXX}
CXXFLAGS+=-Wall -Wextra -Werror -pedantic -std=c++11  -gpu-architecture=compute_70 --gpu-code=sm_70 --use_fast_math -O3
LDFLAGS+=-lm $(CXXFLAGS)

OBJS=cg_main.o cg.o matrix.o matrix_coo.o mmio.o

all: cgsolver

cgsolver: $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

clean:
	rm -f cgsolver *.o *~
