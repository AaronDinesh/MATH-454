OPTIM+=-O3

# CXX=mpicxx
# CC=mpicxx
CXX=mpiicpc
CC=mpiicc

LD=${CXX}
CXXFLAGS+=-Wall -Wextra -std=c++11 $(OPTIM)
# LDFLAGS+=$(OPTIM) -lm
LDFLAGS+=$(OPTIM)

EXECS=pi pi_p2p_ring pi_gather pi_p2p_async_ring pi_p2p_sendrecv_ring pi_reduce

all: clean $(EXECS)

clean:
	rm -f $(EXECS) *.o *~
