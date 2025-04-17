# CXX=g++
CXX=mpicxx
LD=${CXX}
CXXFLAGS+=-O3 -Wall -Wextra -Werror -pedantic -std=c++11 -I${OPENBLAS_ROOT}/include
LDFLAGS+=-lm $(CXXFLAGS) -L${OPENBLAS_ROOT}/lib -lopenblas

OBJS=cg_main.o cg.o matrix_coo.o mmio.o

all: cgsolver

cgsolver: $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

clean:
	rm -f cgsolver *.o *~
