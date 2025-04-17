PHPC - CONJUGATE GRADIENT PROJECT

HOWTO COMPILE AND RUN
=====================

Requirements : 

- a recent compiler (like gcc or intel)
- a cblas library (like openblas or intel MKL)

compile on SCITAS clusters :

```
$ module load gcc openblas
$ make
```

You should see this output (timing is indicative) :

```
$ srun ./cgsolver_sol2 lap2D_5pt_n100.mtx 
size of matrix = 10000 x 10000
Call cgsolver() on matrix size (10000 x 10000)
	[STEP 488] residual = 1.103472E-10
Time for CG = 36.269389 [s]
```

The given example is a 5-points stencil for the 2D Laplace problem. The matrix is in sparse format.

The matrix format is [Matrix Market format (.mtx extension)](https://sparse.tamu.edu/). You can use other matrices there or create your own. 


SOLUTION
========

This solution is a refined version of solution 2. As before, the matrix is partitioned by rows,
not by entries. This is done at the level of the matrix reading.

In addition to the parallel matrix vector product already introduced in section 2,
in this refined version we also consider that operations as vector addition and
scalar product are going to be performed in parallel. So, the processes keep just pieces of
the vectors (except p and x). Thus, in the case of scalar products, each
process performs its part of the calculation, and then call MPI_Allreduce to sum all the
contributions together and communicate them to all the processes.

Beyond this we also need to communicate the vector p at the end of each iteration (to then
perform the matrix vector product), and the x vector once the solution has converged.
For perfoming this communications we use MPI_Allgatherv: each process sends its piece of vector to
every other process, and thus all the process create a reconstruction of the full vector p (and x).


RESULTS IN JED
==============

Without optimization (-O0) and using the matrix lap2D_5pt_n600.mtx, the obtained results in jed are

ntasks        time (s)    speedup
---------------------------------
1             85.055      1
2             44.4383     1.91
4             24.2259     3.51
8             15.0053     5.67
16            9.74829     8.72
32            11.2523     7.55
64            15.7877     5.38


Without optimization (-O0) and using the matrix lap2D_5pt_n1000.mtx, the obtained results in jed are

ntasks        time (s)    speedup
---------------------------------
1             390.156     1
2             198.324     1.97
4             106.302     3.67
8             70.4412     5.54
16            41.3869     9.43
32            48.6894     8.01
64            75.9248     5.14

With optimization (-O3) and using the matrix lap2D_5pt_n600.mtx, the obtained results in jed are

ntasks        time (s)    speedup
---------------------------------
1             18.3274     1
2             11.3259     1.61
4             5.51656     3.23
8             6.26642     2.92
16            9.26687     1.98
32            7.79544     2.35
64            14.4391     1.27


With optimization (-O3) and using the matrix lap2D_5pt_n1000.mtx, the obtained results in jed are

ntasks        time (s)    speedup
---------------------------------
1             105.216     1
2             54.3428     1.93
4             44.0501     2.39
8             41.5415     2.53
16            32.1451     3.27
32            42.9582     2.45
64            71.5748     1.47


Using a dense matrix (same type of matrix, but keeping the zeros) lap2D_5pt_n75_dense.mtx, with -O3 the obtained results in jed are

ntasks        time (s)    speedup
---------------------------------
1             544.191     1
2             340.749     1.60
4             186.901     2.91
8             113.045     4.81
16            55.1272     9.87
32            40.6133     13.40
64            27.0502     20.11

scaling is improving ... Likely, using a larger matrix things will improve further.