#include "cg.hh"
#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <cassert>

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
  int size, rank;

  //Get the size and rank
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  std::vector<int> local_mn_counts(size);
  std::vector<int> local_mn_displacements(size);



  int base_count = m_n / size;
  int local_mn_remainder = m_n % size;

  //Compute the displacement and the counts for each processor
  for (int i = 0; i < size; ++i) {
    local_mn_counts[i] = base_count + (i < local_mn_remainder ? 1 : 0);
    local_mn_displacements[i] = (i == 0) ? 0 : local_mn_displacements[i - 1] + local_mn_counts[i - 1];
  }

    
  std::vector<double> r_global(m_n);
  std::vector<double> p_global(m_n);
  std::vector<double> Ap_global(m_n);
  std::vector<double> tmp(m_n);
  std::vector<double> Ap_local(local_mn_counts[rank]);
  std::vector<double> mat_vec_partial_product(m_n);
  // r = b - A * x;
  m_A.mat_vec(x, mat_vec_partial_product);
  
  //?Reduce then we take a chunk and start processing stuff locally 
  //Reduce to compute the full mat vec product and then scatter the result to Ap_local
  MPI_Reduce_scatter(mat_vec_partial_product.data(), Ap_local.data(), local_mn_counts.data(), MPI_DOUBLE, MPI_SUM, comm);
 
  //Access only my portion of m_b into r_local
  // r_local = m_b[displacements[rank]:displacements[rank] + counts[rank]];
  std::vector<double> r_local(m_b.begin() + local_mn_displacements[rank], m_b.begin() + local_mn_displacements[rank] + local_mn_counts[rank]);
  
  //Now use BLAS to do b - A * x
  cblas_daxpy(local_mn_counts[rank], -1., Ap_local.data(), 1, r_local.data(), 1);



  // MPI_Allreduce(Ap_local.data(), Ap_global.data(), m_n, MPI_DOUBLE, MPI_SUM, comm);
  // r_global = m_b;
  // cblas_daxpy(m_n, -1., Ap_global.data(), 1, r_global.data(), 1);

  // p = r;
  //p_global = r_global;
  MPI_Allgatherv(r_local.data(), local_mn_counts[rank], MPI_DOUBLE, p_global.data(), local_mn_counts.data(), local_mn_displacements.data(), MPI_DOUBLE, comm);
  r_global = p_global;
  
  //! CODE WORKS UNTIL THIS COMMENT

  // rsold = r' * r;
  double rsold_global = cblas_ddot(p_global.size(), p_global.data(), 1, p_global.data(), 1);
  // for i = 1:length(b)
  int k = 0;
  for (; k < m_n; ++k)
  {
    // Ap = A * p;
    m_A.mat_vec(p_global, mat_vec_partial_product);

    MPI_Allreduce(mat_vec_partial_product.data(), Ap_global.data(), m_n, MPI_DOUBLE, MPI_SUM, comm);


    //MPI_Reduce_scatter(mat_vec_partial_product.data(), Ap_local.data(), local_mn_counts.data(), MPI_DOUBLE, MPI_SUM, comm);
    // double dot_prod_result = cblas_ddot(local_mn_counts[rank], Ap_local.data(), 1, p_global.data(), 1);
    // MPI_Allreduce(&dot_prod_result, &dot_prod_result, 1, MPI_DOUBLE, MPI_SUM, comm);

    double dot_prod_result = cblas_ddot(m_n, Ap_global.data(), 1, p_global.data(), 1);
    
    
    // alpha = rsold / (p' * Ap);
    //auto alpha = rsold_global / std::max(cblas_ddot(m_n, p_global.data(), 1, Ap_global.data(), 1), rsold_global * NEARZERO);
    double alpha = rsold_global / std::max(dot_prod_result, rsold_global * NEARZERO);

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

    auto beta = rsnew / rsold_global;
    // p = r + (rsnew / rsold) * p;
    tmp = r_global;
    cblas_daxpy(m_n, beta, p_global.data(), 1, tmp.data(), 1);
    p_global = tmp;

    // rsold = rsnew;
    rsold_global = rsnew;
    if (DEBUG && rank == 0)
    {
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold_global) << "\r" << std::flush;
    }
  }

  if (DEBUG)
  { 
    std::vector<double> r_local_debug(m_n);
    std::vector<double> r_global_debug(m_n);
    m_A.mat_vec(x, r_local_debug);
    MPI_Allreduce(r_local_debug.data(), r_global_debug.data(), m_n, MPI_DOUBLE, MPI_SUM, comm);
    cblas_daxpy(m_n, -1., m_b.data(), 1, r_global_debug.data(), 1);
    auto res =
      std::sqrt(cblas_ddot(m_n, r_global_debug.data(), 1, r_global_debug.data(), 1)) / std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
    auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));
    std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold_global) << ", ||x|| = " << nx
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
