CC=icc
FLAGS= -std=gnu99 -g -O3 -xHost
#
FLAGS += -fopenmp
#FLAGS += -DPAPI -I/ssoft/papi/5.4.0/gnu/include/ /ssoft/papi/5.4.0/gnu/lib/libpapi.a
#FLAGS +=  -I/apps/dora/iaca/iaca-lin32/include/
FLAGS += -no-multibyte-chars
#
all: jacobi-naive jacobi-naive-auto-vec jacobi-sse jacobi-avx jacobi-avx-peel jacobi-avx-block
#
jacobi-naive: jacobi.o
	$(CC) -c $(FLAGS) -no-vec $@.c -o $@.o
	$(CC) -S $(FLAGS) -no-vec $@.c -o $@.s 
	$(CC) $(FLAGS) -no-vec $^ $@.o -o $@
#
jacobi-naive-auto-vec: jacobi.o
	$(CC) -c $(FLAGS) jacobi-naive.c -o $@.o
	$(CC) -S $(FLAGS) jacobi-naive.c -o $@.s 
	$(CC) $(FLAGS) -no-vec $^ $@.o -o $@
#
jacobi-sse: jacobi.o jacobi-sse.o
	$(CC) $(FLAGS) $^ -o $@
#
jacobi-avx: jacobi.o jacobi-avx.o
	$(CC) $(FLAGS) $^ -o $@
#
jacobi-avx-peel: jacobi.o jacobi-avx-peel.o
	$(CC) $(FLAGS) $^ -o $@
#
jacobi-avx-block: jacobi.o jacobi-avx-block.o
	$(CC) $(FLAGS) $^ -o $@
#
%.o: %.c
	$(CC) -c $(FLAGS) $<
	$(CC) -S $(FLAGS) $< -o $@.s 
#
clean:
	rm -f *.s *.lst *.o jacobi-avx jacobi-avx-peel jacobi-avx-block jacobi-sse jacobi-naive jacobi-naive-auto-vec
