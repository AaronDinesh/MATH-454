#include "cg.hh"
#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>
#include <mpi.h>

const double NEARZERO = 1.0e-14;
const bool DEBUG = true;

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
CGSolverSparse::solve(std::vector<double> &x, MPI_Comm comm){
  std::vector<double> r_global(m_n);
  std::vector<double> p_global(m_n);
  std::vector<double> Ap_global(m_n);
  std::vector<double> tmp(m_n);
  std::vector<double> Ap_local(m_n);
  
  // r = b - A * x;
  m_A.mat_vec(x, Ap_local);
  
  //? Insteadd of reducing here we can keep Ap distributed locally and then only reduce r
  MPI_Allreduce(Ap_local.data(), Ap_global.data(), m_n, MPI_DOUBLE, MPI_SUM, comm);
  
  // std::vector<double> r_local(m_n);
  // r_local = m_b;
  // cblas_daxpy(m_n, -1., Ap_local.data(), 1, r_local.data(), 1); 
  // MPI_Allreduce(r_local.data(), r_global.data(), m_n, MPI_DOUBLE, MPI_SUM, comm);

  //? Instead of computing r on all processes we can call daxpy on all processes and then reduce.
  r_global = m_b;
  cblas_daxpy(m_n, -1., Ap_global.data(), 1, r_global.data(), 1);

  // p = r;
  p_global = r_global;

  // rsold = r' * r;
  auto rsold = cblas_ddot(m_n, r_global.data(), 1, r_global.data(), 1);

  // for i = 1:length(b)
  int k = 0;
  for (; k < m_n; ++k)
  {
    // Ap = A * p;
    m_A.mat_vec(p_global, Ap_local);

    MPI_Allreduce(Ap_local.data(), Ap_global.data(), m_n, MPI_DOUBLE, MPI_SUM, comm);

    // alpha = rsold / (p' * Ap);
    auto alpha = rsold / std::max(cblas_ddot(m_n, p_global.data(), 1, Ap_global.data(), 1), rsold * NEARZERO);

    // x = x + alpha * p;
    cblas_daxpy(m_n, alpha, p_global.data(), 1, x.data(), 1);

    // r = r - alpha * Ap;
    cblas_daxpy(m_n, -alpha, Ap_global.data(), 1, r_global.data(), 1);

    // rsnew = r' * r;
    auto rsnew = cblas_ddot(m_n, r_global.data(), 1, r_global.data(), 1);

    // if sqrt(rsnew) < 1e-10
    //   break;
    if (std::sqrt(rsnew) < m_tolerance)
      break; // Convergence test

    auto beta = rsnew / rsold;
    // p = r + (rsnew / rsold) * p;
    tmp = r_global;
    cblas_daxpy(m_n, beta, p_global.data(), 1, tmp.data(), 1);
    p_global = tmp;

    // rsold = rsnew;
    rsold = rsnew;
    if (DEBUG)
    {
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold) << "\r" << std::flush;
    }
  }

  if (DEBUG)
  { 
    std::vector<double> r_local(m_n);
    m_A.mat_vec(x, r_local);
    MPI_Allreduce(r_local.data(), r_global.data(), m_n, MPI_DOUBLE, MPI_SUM, comm);
    cblas_daxpy(m_n, -1., m_b.data(), 1, r_global.data(), 1);
    auto res =
      std::sqrt(cblas_ddot(m_n, r_global.data(), 1, r_global.data(), 1)) / std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
    auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));
    std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold) << ", ||x|| = " << nx
              << ", ||Ax - b||/||b|| = " << res << std::endl;
  }
}

void CGSolverSparse::read_matrix(const std::string &filename){
  m_A.read(filename);
  m_m = m_A.m();
  m_n = m_A.n();
}

void CGSolverSparse::read_matrix_distributed(const std::string& filename, MPI_Comm comm){
  m_A.read_distributed(filename, comm);
  m_m = m_A.m();
  m_n = m_A.n();
}

/*
Initialization of the source term b
*/
void Solver::init_source_term(double h){
  m_b.resize(m_n);

  for (int i = 0; i < m_n; i++)
  {
    m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) * std::sin(10. * M_PI * i * h);
  }
}
