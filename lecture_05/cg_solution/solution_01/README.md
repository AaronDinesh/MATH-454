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
$ srun ./cgsolver lap2D_5pt_n100.mtx 
size of matrix = 10000 x 10000
Call cgsolver() on matrix size (10000 x 10000)
	[STEP 488] residual = 1.103472E-10
Time for CG = 36.269389 [s]
```

The given example is a 5-points stencil for the 2D Laplace problem. The matrix is in sparse format.

The matrix format is [Matrix Market format (.mtx extension)](https://sparse.tamu.edu/). You can use other matrices there or create your own. 


RESULTS IN JED
==============

In this solution we just partition the matrix according to its entries, not its rows.
This is done when the matrix is parsed.
Then, we assume that the total matrix A can be written as A=A1 + A2 + ... + Ap, where
Ai are the matrices parts owned by the p different MPI proccess.
When we compute A * x, we actually compute the vectors y1 = A1 * x, y2 = A2 * x, ..., yp = Ap * x
using different processes, and then we have to sum the together: A * x = y1 + y2 + ... + yp
Because the way the matrix is partitioned, we cannot assume that the different vectors yi
are disjoint (in the sense that every vector has only non zero values in a set of rows that
do not intersect with any other vector). Instead, it is possible that the different vectors
yi have values in the coincident rows.
This fact prevents from just from simply doing an MPI_Allgather communication and instead
we have to perform an MPI_Allreduce.


RESULTS
========

Without optimization (-O0) and using the matrix lap2D_5pt_n600.mtx, the obtained results in jed are

ntasks        time (s)    speedup
---------------------------------
1             67.6614     1
2             46.6        1.45
4             38.30       1.76
8             32.00       2.11
16            40.02       1.69


Without optimization (-O0) and using the matrix lap2D_5pt_n1000.mtx, the obtained results in jed are

ntasks        time (s)    speedup
---------------------------------
1             327.18      1
2             215.61      1.51
4             185.09      1.77
8             191.53      1.71
16            201.14      1.63 

With optimization (-O3) and using the matrix lap2D_5pt_n600.mtx, the obtained results in jed are

ntasks        time (s)    speedup
---------------------------------
1             20.96       1
2             21.10       0.99
4             23.66       0.86
8             30.93       0.68

With optimization (-O3) and using the matrix lap2D_5pt_n1000.mtx, the obtained results in jed are

ntasks        time (s)    speedup
---------------------------------
1             109.67      1
2             77.43       1.42
4             113.89      0.96
8             127.54      0.86
