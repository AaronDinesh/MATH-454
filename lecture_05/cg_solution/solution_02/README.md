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


RESULTS IN JED
==============

In this solution we just partition the matrix according to its rows, not its entries.
This is done when the matrix is parsed.
In the case of matrices with heterogeneous sparsity patterns this could lead to an unbalanced
partitioning. Not the case in the exercise

Then, we assume that the total matrix A can be written as A=A1 + A2 + ... + Ap, where
Ai are the matrices parts owned by the p different MPI proccess.
When we compute A * x, we actually compute the vectors y1 = A1 * x, y2 = A2 * x, ..., yp = Ap * x
using different processes, and then we have to sum the together: A * x = y1 + y2 + ... + yp
Because the the matrix is partitioned by its rows, we can assume that the different vectors yi
are disjoint (in the sense that every vector has only non zero values in a set of rows that
do not intersect with any other vector).
This fact allows us to simply perform a MPI_Allgather(v) communication to sum all the vectors.


RESULTS
========

Without optimization (-O0) and using the matrix lap2D_5pt_n600.mtx, the obtained results in jed are

ntasks        time (s)    speedup
---------------------------------
1             76.41       1
2             52.11       1.47
4             30.48       2.50
8             26.98       2.83
16            26.25       2.91


Without optimization (-O0) and using the matrix lap2D_5pt_n1000.mtx, the obtained results in jed are

ntasks        time (s)    speedup
---------------------------------
1            434.81       1
2            251.66       1.73
4            178.12       2.44
8            152.90       2.84
16           142.384      3.05 
32           142.384      3.05 

With optimization (-O3) and using the matrix lap2D_5pt_n600.mtx, the obtained results in jed are

ntasks        time (s)    speedup
---------------------------------

With optimization (-O3) and using the matrix lap2D_5pt_n1000.mtx, the obtained results in jed are

ntasks        time (s)    speedup
---------------------------------
