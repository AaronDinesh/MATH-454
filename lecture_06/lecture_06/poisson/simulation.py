from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD


class Simulation():


    def __init__(self, n_start, n_loc, n, n_iter) -> None:
        self.n_start = n_start
        self.n_loc = n_loc
        self.n = n
        self.n_iter = n_iter
        self.h_n = self.h_m = 1/self.n
        self.f = lambda x, y: - 200 * np.sin(10*np.pi*x) * np.sin(10*np.pi*y)
        self.x, self.y = np.meshgrid(np.linspace(0, 1, self.n), np.linspace(0, 1, self.n))
        self.x, self.y = self.x[n_start:n_start+n_loc, :], self.y[n_start:n_start+n_loc, :]
        self.u, self.u0 = np.zeros((self.n_loc, self.n)), np.zeros((self.n_loc, self.n))

    def set_initial_conditions(self):
        self.f_ = self.f(self.x, self.y)

    def compute(self, sync):
        s = 0
        while (s < self.n_iter):
            l2 = self.compute_step(sync)
            self.u, self.u0 = self.u0, self.u
            s += 1

        return l2, s
    
    def compute_step(self, sync):
        self.comm = MPI.COMM_WORLD
        next_prank = self.comm.rank + 1
        if self.comm.rank == (self.comm.size - 1):
            next_prank = MPI.PROC_NULL

        prev_prank = self.comm.rank - 1
        if self.comm.rank == 0:
            prev_prank = MPI.PROC_NULL

        tag = 0
        
        if sync == 'sync':  # synchronous version
            self.comm.Sendrecv(self.u0[1, :], prev_prank, tag, self.u0[self.n_loc-1, :], next_prank, tag)
            self.comm.Sendrecv(self.u0[self.n_loc-2, 0], next_prank, tag, self.u0[0, :], prev_prank, tag)

            self.u[1:self.n_loc-1, 1:-1] = 0.25 * (self.u0[:self.n_loc-2, 1:-1] + self.u0[2:, 1:-1] + self.u0[1:self.n_loc-1, 0:-2] +
                                                   self.u0[1:self.n_loc-1, 2:] - self.f_[1:self.n_loc-1, 1:-1] * self.h_m * self.h_n)
            l2 = np.linalg.norm(self.u[1:self.n_loc-1, 1:-1]-self.u0[1:self.n_loc-1, 1:-1]) ** 2
        else:  # asynchronous version
            requests = [None] * 4
            requests[0] = self.comm.Irecv(self.u0[0, :], prev_prank, tag)
            requests[1] = self.comm.Irecv(self.u0[self.n_loc-1, :], next_prank, tag)

            requests[2] = self.comm.Isend(self.u0[1, :], prev_prank, tag)
            requests[3] = self.comm.Isend(self.u0[self.n_loc-2, :], next_prank, tag)

            self.u[2:self.n_loc-2, 1:-1] = 0.25 * (self.u0[1:self.n_loc-3, 1:-1] + self.u0[3:self.n_loc-1, 1:-1] + self.u0[2:self.n_loc-2, 0:-2] +
                                                   self.u0[2:self.n_loc-2, 2:] - self.f_[2:self.n_loc-2, 1:-1] * self.h_m * self.h_n)
            l2 = np.linalg.norm(self.u[2:self.n_loc-2, 1:-1]-self.u0[2:self.n_loc-2, 1:-1]) ** 2

            requests[0].wait()
            requests[1].wait()
            requests[2].wait()
            requests[3].wait()

            self.u[1:2, 1:-1] = 0.25 * (self.u0[0:1, 1:-1] + self.u0[2:3, 1:-1] + self.u0[1:2, 0:-2] +
                                        self.u0[1:2, 2:] - self.f_[1:2, 1:-1] * self.h_m * self.h_n)
            self.u[self.n_loc-2:self.n_loc-1, 1:-1] = 0.25 * (self.u0[self.n_loc-3:self.n_loc-2, 1:-1] + self.u0[self.n_loc-1:self.n_loc, 1:-1] + self.u0[self.n_loc-2:self.n_loc-1, 0:-2] +
                                                              self.u0[self.n_loc-2:self.n_loc-1, 2:] - self.f_[self.n_loc-2:self.n_loc-1, 1:-1] * self.h_m * self.h_n)

            l2 += np.linalg.norm(self.u[1:2, 1:-1]-self.u0[1:2, 1:-1]) ** 2
            l2 += np.linalg.norm(self.u[self.n_loc-2:self.n_loc-1, 1:-1]-self.u0[self.n_loc-2:self.n_loc-1, 1:-1]) ** 2

        l2 = comm.allreduce(l2, MPI.SUM)

        return l2