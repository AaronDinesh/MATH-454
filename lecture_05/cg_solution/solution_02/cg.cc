#include "cg.hh"

#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>
#include <mpi.h>

const double NEARZERO = 1.0e-14;
const bool DEBUG = false;

/*
    cgsolver solves the linear equation A*x = b where A is
    of size m x n

Code based on MATLAB code (from wikipedia ;-)  ):

function x = conjgrad(A, b, x)
    r = b - A * x;
    p = r;
    rsold = r' * r;

    for i = 1:length(b)
        Ap = A * p;
        alpha = rsold / (p' * Ap);
        x = x + alpha * p;
        r = r - alpha * Ap;
        rsnew = r' * r;
        if sqrt(rsnew) < 1e-10
              break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end

*/

void
CGSolverSparse::communicate_vector(const std::vector<double> &local_vector,
                                   std::vector<double> &global_vector) const
{
  int psize;
  MPI_Comm_size(MPI_COMM_WORLD, &psize);

  const int m_local = m_A.m() / psize;
  std::vector<int> counts(psize, m_local);
  counts.back() = m_A.m() - m_local * (psize - 1);

  std::vector<int> disp(psize);
  disp[0] = 0;
  for(int i = 1; i < psize; ++i)
  {
    disp[i] = disp[i-1] + counts[i-1];
  }

  global_vector.resize(disp.back() + counts.back());

  MPI_Allgatherv(local_vector.data() + m_A.m_start(), m_A.m_local(), MPI_DOUBLE, 
                 global_vector.data(), counts.data(), disp.data(), MPI_DOUBLE, MPI_COMM_WORLD);

}

/*
Sparse version of the cg solver
*/
void
CGSolverSparse::solve(std::vector<double> &x)
{
  int prank, psize;
  MPI_Comm_rank(MPI_COMM_WORLD, &prank);
  MPI_Comm_size(MPI_COMM_WORLD, &psize);


  static_cast<void>(x);
  std::vector<double> r(m_n);
  std::vector<double> p(m_n);
  std::vector<double> Ap(m_n);
  std::vector<double> tmp(m_n);

  // r = b - A * x;
  m_A.mat_vec(x, Ap);
  communicate_vector(Ap, Ap);

  r = m_b;
  cblas_daxpy(m_n, -1., Ap.data(), 1, r.data(), 1);

  // p = r;
  p = r;

  // rsold = r' * r;
  auto rsold = cblas_ddot(m_n, r.data(), 1, r.data(), 1);

  // for i = 1:length(b)
  int k = 0;
  for (; k < m_n; ++k)
  {
    // Ap = A * p;
    m_A.mat_vec(p, Ap);
    communicate_vector(Ap, Ap);

    // alpha = rsold / (p' * Ap);
    auto alpha = rsold / std::max(cblas_ddot(m_n, p.data(), 1, Ap.data(), 1), rsold * NEARZERO);

    // x = x + alpha * p;
    cblas_daxpy(m_n, alpha, p.data(), 1, x.data(), 1);

    // r = r - alpha * Ap;
    cblas_daxpy(m_n, -alpha, Ap.data(), 1, r.data(), 1);

    // rsnew = r' * r;
    auto rsnew = cblas_ddot(m_n, r.data(), 1, r.data(), 1);

    // if sqrt(rsnew) < 1e-10
    //   break;
    if (std::sqrt(rsnew) < m_tolerance)
      break; // Convergence test

    auto beta = rsnew / rsold;
    // p = r + (rsnew / rsold) * p;
    tmp = r;
    cblas_daxpy(m_n, beta, p.data(), 1, tmp.data(), 1);
    p = tmp;

    // rsold = rsnew;
    rsold = rsnew;
    if (DEBUG)
    {
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold) << "\r" << std::flush;
    }
  }

  if (prank == 0) {
    std::cout << "#iterations: " << k << std::endl;
  }

  if (DEBUG)
  {
  m_A.mat_vec(x, r);
  cblas_daxpy(m_n, -1., m_b.data(), 1, r.data(), 1);
  auto res =
    std::sqrt(cblas_ddot(m_n, r.data(), 1, r.data(), 1)) / std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
  auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));
  std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold) << ", ||x|| = " << nx
            << ", ||Ax - b||/||b|| = " << res << std::endl;
  }
}

void
CGSolverSparse::read_matrix(const std::string &filename)
{
  m_A.read(filename);
  m_m = m_A.m();
  m_n = m_A.n();
}

/*
Initialization of the source term b
*/
void
Solver::init_source_term(double h)
{
  m_b.resize(m_n);

  for (int i = 0; i < m_n; i++)
  {
    m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) * std::sin(10. * M_PI * i * h);
  }
}
