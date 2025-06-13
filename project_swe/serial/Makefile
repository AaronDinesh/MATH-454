CXX=g++
LD=${CXX}
CXXFLAGS+= -Wall -Wextra -pedantic -std=c++11 -I${HDF5_ROOT}/include -O1 -g -fno-omit-frame-pointer
LDFLAGS+=-lm $(CXXFLAGS) -L${HDF5_ROOT}/lib -lhdf5 

OBJS= main.o swe.o xdmf_writer.o

all: swe

swe: $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

clean:
	rm -f swe xdmf *.o *~
