#include "cg.hh"
#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <cassert>

const double NEARZERO = 1.0e-14;
//const bool DEBUG = true;
#define DEBUG
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
  std::vector<double> r_local(local_mn_counts[rank]); 


  std::vector<double> p_global(m_n);
  std::vector<double> p_local(local_mn_counts[rank]);

  std::vector<double> Ap_global(m_n);
  std::vector<double> tmp_local(local_mn_counts[rank]);
  std::vector<double> Ap_local(local_mn_counts[rank]);
  std::vector<double> mat_vec_partial_product(m_n);
  // r = b - A * x;
  m_A.mat_vec(x, mat_vec_partial_product);
  
  //?Reduce then we take a chunk and start processing stuff locally 
  //Reduce to compute the full mat vec product and then scatter the result to Ap_local
  MPI_Reduce_scatter(mat_vec_partial_product.data(), Ap_local.data(), local_mn_counts.data(), MPI_DOUBLE, MPI_SUM, comm);
 
  //Access only my portion of m_b into r_local
  // r_local = m_b[displacements[rank]:displacements[rank] + counts[rank]];
  cblas_dcopy(local_mn_counts[rank], m_b.data() + local_mn_displacements[rank], 1, r_local.data(), 1);
  
  //Now use BLAS to do b - A * x
  cblas_daxpy(local_mn_counts[rank], -1., Ap_local.data(), 1, r_local.data(), 1);

  cblas_dcopy(local_mn_counts[rank], r_local.data(), 1, p_local.data(), 1);

  // rsold = r' * r;
  double rsold_local = cblas_ddot(local_mn_counts[rank], r_local.data(), 1, r_local.data(), 1);  
  double rsold_global;
  MPI_Request request1;
  
  MPI_Iallreduce(&rsold_local, &rsold_global, 1, MPI_DOUBLE, MPI_SUM, comm, &request1);
  
  #ifdef DEBUG
    MPI_Request request2;
  #endif 
  // for i = 1:length(b)
  int k = 0;
  for (; k < m_n; ++k){
    MPI_Allgatherv(p_local.data(), local_mn_counts[rank], MPI_DOUBLE, p_global.data(), local_mn_counts.data(), local_mn_displacements.data(), MPI_DOUBLE, comm);
    
    // Ap = A * p;
    m_A.mat_vec(p_global, mat_vec_partial_product);
    MPI_Reduce_scatter(mat_vec_partial_product.data(), Ap_local.data(), local_mn_counts.data(), MPI_DOUBLE, MPI_SUM, comm);
    
    
    double local_dot_prod_result = cblas_ddot(local_mn_counts[rank], Ap_local.data(), 1, p_local.data(), 1);
    double global_dot_prod_result;
    MPI_Allreduce(&local_dot_prod_result, &global_dot_prod_result, 1, MPI_DOUBLE, MPI_SUM, comm);
    
    // alpha = rsold / (p' * Ap);
    //auto alpha = rsold_global / std::max(cblas_ddot(m_n, p_global.data(), 1, Ap_global.data(), 1), rsold_global * NEARZERO);
    MPI_Wait(&request1, MPI_STATUS_IGNORE);
    double alpha = rsold_global / std::max(global_dot_prod_result, rsold_global * NEARZERO);
    
    // x = x + alpha * p;
    cblas_daxpy(local_mn_counts[rank], alpha, p_local.data(), 1, &x[local_mn_displacements[rank]], 1);
    
    #ifdef DEBUG
      MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x.data(), local_mn_counts.data(), local_mn_displacements.data(), MPI_DOUBLE, comm, &request2);
    #endif
    // r = r - alpha * Ap;
    cblas_daxpy(local_mn_counts[rank], -alpha, Ap_local.data(), 1, r_local.data(), 1);

    // rsnew = r' * r;
    double rsnew_local = cblas_ddot(local_mn_counts[rank], r_local.data(), 1, r_local.data(), 1);
    double rsnew_global;

    MPI_Request request3;
    MPI_Iallreduce(&rsnew_local, &rsnew_global, 1, MPI_DOUBLE, MPI_SUM, comm, &request3);
    cblas_dcopy(local_mn_counts[rank], r_local.data(), 1, tmp_local.data(), 1);
    
    // if sqrt(rsnew) < 1e-10
    //   break;
    
    MPI_Wait(&request3, MPI_STATUS_IGNORE);
    if (std::sqrt(rsnew_global) < m_tolerance)
      break; // Convergence test


    double beta = rsnew_global / rsold_global;
    // p = r + (rsnew / rsold) * p;
    cblas_daxpy(local_mn_counts[rank], beta, p_local.data(), 1, tmp_local.data(), 1);
    p_local = tmp_local;
    rsold_global = rsnew_global;

    #ifdef DEBUG
    if (rank == 0){
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold_global) << "\r" << std::flush;
    }
    #endif

  }
  
  #ifdef DEBUG
    std::vector<double> r_local_debug(m_n);
    std::vector<double> r_global_debug(m_n);
    MPI_Wait(&request2, MPI_STATUS_IGNORE);
    m_A.mat_vec(x, r_local_debug);
    MPI_Allreduce(r_local_debug.data(), r_global_debug.data(), m_n, MPI_DOUBLE, MPI_SUM, comm);
    cblas_daxpy(m_n, -1., m_b.data(), 1, r_global_debug.data(), 1);
    double res = std::sqrt(cblas_ddot(m_n, r_global_debug.data(), 1, r_global_debug.data(), 1)) / std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
    double nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));
    std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold_global) << ", ||x|| = " << nx
              << ", ||Ax - b||/||b|| = " << res << std::endl;
  #endif
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
