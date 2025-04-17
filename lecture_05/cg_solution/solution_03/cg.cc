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
  int psize, prank;
  MPI_Comm_size(MPI_COMM_WORLD, &psize);
  MPI_Comm_rank(MPI_COMM_WORLD, &prank);

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

  MPI_Allgatherv(local_vector.data(), counts[prank], MPI_DOUBLE, 
                 global_vector.data(), counts.data(), disp.data(), MPI_DOUBLE, MPI_COMM_WORLD);

}

/*
Sparse version of the cg solver
*/
void
CGSolverSparse::solve(std::vector<double> &x_global)
{
  const int m_local = m_A.m_local();
  std::vector<double> x(m_local, 0.0);
  std::vector<double> r(m_local);
  std::vector<double> p(m_local);
  std::vector<double> p_global(m_n);
  std::vector<double> Ap(m_local);
  std::vector<double> tmp(m_local);

  int prank;
  MPI_Comm_rank(MPI_COMM_WORLD, &prank);

  // r = b - A * x;
  m_A.mat_vec(x_global, Ap);
  for(int i = 0; i < m_local; ++i)
  {
    // r = m_b;
    r[i] = m_b[m_A.m_start() + i];
  }

  cblas_daxpy(m_local, -1., Ap.data(), 1, r.data(), 1);

  // p = r;
  p = r;
  communicate_vector(p, p_global);

  // rsold = r' * r = p' * p;
  // auto rsold = cblas_ddot(m_n, p_global.data(), 1, p_global.data(), 1);
  auto rsold_local = cblas_ddot(m_local, r.data(), 1, r.data(), 1);
    double rsold;
    MPI_Allreduce(&rsold_local, &rsold, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // for i = 1:length(b)
  int k = 0;
  for (; k < m_n; ++k)
  {
    // Ap = A * p;
    m_A.mat_vec(p_global, Ap);
    const auto pAp_local = cblas_ddot(m_local, p.data(), 1, Ap.data(), 1);
    double pAp;
    MPI_Allreduce(&pAp_local, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // alpha = rsold / (p' * Ap);
    auto alpha = rsold / std::max(pAp, rsold * NEARZERO);

    // x = x + alpha * p;
    cblas_daxpy(m_local, alpha, p.data(), 1, x.data(), 1);

    // r = r - alpha * Ap;
    cblas_daxpy(m_local, -alpha, Ap.data(), 1, r.data(), 1);

    // rsnew = r' * r;
    const auto rsnew_local = cblas_ddot(m_local, r.data(), 1, r.data(), 1);
    double rsnew;
    MPI_Allreduce(&rsnew_local, &rsnew, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // if sqrt(rsnew) < 1e-10
    //   break;
    if (std::sqrt(rsnew) < m_tolerance)
      break; // Convergence test

    auto beta = rsnew / rsold;
    // p = r + (rsnew / rsold) * p;
    tmp = r;
    cblas_daxpy(m_local, beta, p.data(), 1, tmp.data(), 1);
    p = tmp;
    communicate_vector(p, p_global);

    // rsold = rsnew;
    rsold = rsnew;
    if (DEBUG && prank == 0)
    {
      // std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold) << "\r" << std::flush;
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold) << "\r" << std::endl;
    }
  }

  communicate_vector(x, x_global);

  if (prank == 0) {
    std::cout << "#iterations: " << k << std::endl;
  }

  if (DEBUG)
  {
    m_A.mat_vec(x_global, r);

    std::vector<double> r_global(m_n);
    communicate_vector(r, r_global);
    cblas_daxpy(m_n, -1., m_b.data(), 1, r_global.data(), 1);
    auto res =
      std::sqrt(cblas_ddot(m_n, r_global.data(), 1, r_global.data(), 1)) / std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
    auto nx = std::sqrt(cblas_ddot(m_n, x_global.data(), 1, x_global.data(), 1));
    if (prank == 0)
    {
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold) << ", ||x|| = " << nx
                << ", ||Ax - b||/||b|| = " << res << std::endl;
    }
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
