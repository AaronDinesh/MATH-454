PHPC - CONJUGATE GRADIENT PROJECT

HOWTO COMPILE AND RUN
=====================

## Serial Compilation
1) Navigate to the serial directory.
2) Run the command ``make clean``
3) Run the command ``make``
4) Navigate back to the root directory and run ``./serial/cgsolver <path-to-matrix-file>``

## Parallel Compilation
1) Navigate to the serial directory.
2) Run the command ``make clean``
3) Run the command ``make CXX=mpicxx``
4) Navigate back to the root directory and run ``mpiexec -np 4 ./parallel/cgsolver <path-to-matrix-file>``

## On the cluser
1) Compile the script similarly to the instructions above
2) Run ``sbatch parallel.batch`` with the right parameters
