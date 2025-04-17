OPTIM+=-O3

CXX=mpicxx
CC=mpicxx
LD=${CXX}
CXXFLAGS+=-Wall -Wextra -std=c++11 $(OPTIM)
LDFLAGS+=$(OPTIM) -lm

EXECS=pi_gather pi_p2p_async_ring

all: clean $(EXECS)

clean:
	rm -f $(EXECS) *.o *~
