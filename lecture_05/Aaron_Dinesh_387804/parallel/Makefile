# Default compiler
CXX ?= g++
LD = ${CXX}

# Detect if we're using Intel compiler (e.g., mpiicc)
ifeq ($(findstring mpiicc, $(CXX)), mpiicc)
    # Source the Intel environment first in your terminal!
    # Assume MKLROOT is set
    CXXFLAGS += -Wall -Wextra -Werror -pedantic -std=c++11 -O3 -g
    CXXFLAGS += -DOMPI_SKIP_MPICXX -I${MKLROOT}/include
    LDFLAGS += -Wl,--start-group \
               ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a \
               ${MKLROOT}/lib/intel64/libmkl_sequential.a \
               ${MKLROOT}/lib/intel64/libmkl_core.a \
               -Wl,--end-group -lpthread -lm -ldl
else
    # Assume OpenBLAS setup
    CXXFLAGS += -Wall -Wextra -Werror -pedantic -std=c++11 -I${OPENBLAS_ROOT}/include -O2
    CXXFLAGS += -DOMPI_SKIP_MPICXX
    LDFLAGS += -lm -L${OPENBLAS_ROOT}/lib -lopenblas
endif

OBJS = cg_main.o cg.o matrix_coo.o mmio.o

all: cgsolver

cgsolver: $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

clean:
	rm -f cgsolver *.o *~
