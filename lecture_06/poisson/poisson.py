import numpy as np
import argparse
import time
from simulation import Simulation, comm

# Setup arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', type=int, default=512)
parser.add_argument('-n', type=int, default=512)
parser.add_argument('-n_iter', type=int, default=50)
parser.add_argument('-sync', type=str, default='sync')
args = parser.parse_args()

if comm.rank == 0:
    print("arguments:")
    for arg in vars(args):
        print("\t", arg, getattr(args, arg))

assert args.sync in ['sync', 'async']


def assign_rows_to_processors(N, psize):
    if (psize == 1):
        start_rows, num_rank_cells = [0], [N]
    else:
        start_rows, num_rank_cells = [], []
        N_loc = N // psize
        start_rows.append(0)
        num_rank_cells.append(N_loc)
        i0 = N_loc
        for _ in range(1, psize-1):
            start_rows.append(i0)
            num_rank_cells.append(N_loc)
            i0 += N_loc
        start_rows.append(i0)
        num_rank_cells.append(N-i0)

    return start_rows, num_rank_cells


if __name__ == "__main__":
    start_time = time.time()
    start_rows, num_rank_cells = assign_rows_to_processors(args.m, comm.size)
    simu = Simulation(start_rows[comm.rank], num_rank_cells[comm.rank], args.n, args.n_iter)
    simu.set_initial_conditions()
    l2, s = simu.compute(args.sync)
    if comm.rank == 0:
        print(l2, s, time.time()-start_time)

