CXX=mpic++
LD=${CXX}
CXXFLAGS+=-Wall -Wextra -pedantic -std=c++11 -O3
LDFLAGS+=-lm $(CXXFLAGS)

OBJS=grid.o dumpers.o main.o

all: write_bmp

write_bmp: $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

clean:
	rm -f write_bmp *.o *~
