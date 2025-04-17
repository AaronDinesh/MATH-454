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

/*
Sparse version of the cg solver
*/
void
CGSolverSparse::solve(std::vector<double> &x_global)
{
  std::vector<double> x(m_m_local);
  std::vector<double> r(m_m_local);
  std::vector<double> p(m_m_local);
  std::vector<double> p_global(x_global.size());
  std::vector<double> Ap(m_m_local);
  std::vector<double> tmp(m_m_local);

  // r = b - A * x;
  m_A.mat_vec(x_global, Ap);
  for(int i = m_A.m_start(), j = 0; i < m_A.m_end(); ++i, ++j)
  {
    r[j] = m_b[i];
    x[j] = x_global[i];
  }
  cblas_daxpy(m_m_local, -1., Ap.data(), 1, r.data(), 1);

  p = r;
  communicate_vector(p, p_global);

  // rsold = r' * r;
  auto rsold_local = cblas_ddot(m_m_local, r.data(), 1, r.data(), 1);
  double rsold;
  MPI_Allreduce(&rsold_local, &rsold, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // for i = 1:length(b)
  int k = 0;
  for (; k < m_n; ++k)
  {
    // Ap = A * p;
    m_A.mat_vec(p_global, Ap);

    // alpha = rsold / (p' * Ap);
    auto p_Ap_local = cblas_ddot(m_m_local, p.data(), 1, Ap.data(), 1);
    double p_Ap;
    MPI_Allreduce(&p_Ap_local, &p_Ap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    auto alpha = rsold / std::max(p_Ap, rsold * NEARZERO);

    // x = x + alpha * p;
    cblas_daxpy(m_m_local, alpha, p.data(), 1, x.data(), 1);

    // r = r - alpha * Ap;
    cblas_daxpy(m_m_local, -alpha, Ap.data(), 1, r.data(), 1);

    // rsnew = r' * r;
    auto rsnew_local = cblas_ddot(m_m_local, r.data(), 1, r.data(), 1);
    double rsnew;
    MPI_Allreduce(&rsnew_local, &rsnew, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // if sqrt(rsnew) < 1e-10
    //   break;
    if (std::sqrt(rsnew) < m_tolerance)
      break; // Convergence test

    auto beta = rsnew / rsold;
    // p = r + (rsnew / rsold) * p;
    tmp = r;
    cblas_daxpy(m_m_local, beta, p.data(), 1, tmp.data(), 1);
    p = tmp;
    communicate_vector(p, p_global);

    // rsold = rsnew;
    rsold = rsnew;
    if (DEBUG)
    {
      // std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold) << "\r" << std::flush;
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold) << "\r" << std::endl;
    }
  }

#if 0 // This piece of code is no longer valid, as quantities are local
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
#endif

  communicate_vector(x, x_global);
}

void
CGSolverSparse::read_matrix(const std::string &filename)
{
  m_A.read(filename);
  m_m = m_A.m();
  m_m_local = m_A.m_local();
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

void
CGSolverSparse::communicate_vector(const std::vector<double> &vec_local,
      		                         std::vector<double> &vec_global) const
{
  int prank, psize;
  MPI_Comm_rank(MPI_COMM_WORLD, &prank);
  MPI_Comm_size(MPI_COMM_WORLD, &psize);

  std::vector<int> counts(psize, m_m_local);
  counts.back() = m_m - (psize - 1) * m_m_local;

  std::vector<int> displacements(psize, 0);
  for(int i = 1; i < psize; ++i) {
    displacements[i] = displacements[i-1] + m_m_local;
  }

  MPI_Allgatherv(vec_local.data(), vec_local.size(), MPI_DOUBLE, vec_global.data(), counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
}
