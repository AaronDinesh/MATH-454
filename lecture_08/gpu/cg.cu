#include "config.hh"
#include "cg.hh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include "cuda_ops.hh"
#include <cublas_v2.h>


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
void CGSolver::solve(std::vector<double> & x) {
  std::vector<double> r(m_n);
  std::vector<double> p(m_n);
  std::vector<double> Ap(m_n);
  std::vector<double> tmp(m_n);
  

  // r = b - A * x;
  std::fill_n(Ap.begin(), Ap.size(), 0.);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n,
              x.data(), 1, 0., Ap.data(), 1);

  r = m_b;
  cblas_daxpy(m_n, -1., Ap.data(), 1, r.data(), 1);

  // p = r;
  p = r;

  // rsold = r' * r;
  auto rsold = cblas_ddot(m_n, r.data(), 1, p.data(), 1);

  // for i = 1:length(b)
  int k = 0;
  for (; k < m_n; ++k) {
    // Ap = A * p;
    std::fill_n(Ap.begin(), Ap.size(), 0.);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n,
                p.data(), 1, 0., Ap.data(), 1);

    // alpha = rsold / (p' * Ap);
    auto alpha = rsold / std::max(cblas_ddot(m_n, p.data(), 1, Ap.data(), 1),
                                  rsold * NEARZERO);

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
    #ifdef DEBUG
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                << std::sqrt(rsold) << "\r" << std::flush;
    #endif
  }

  #ifdef DEBUG
    std::fill_n(r.begin(), r.size(), 0.);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n,
                x.data(), 1, 0., r.data(), 1);
    cblas_daxpy(m_n, -1., m_b.data(), 1, r.data(), 1);
    auto res = std::sqrt(cblas_ddot(m_n, r.data(), 1, r.data(), 1)) /
               std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
    auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));
    std::cout << "\t[STEP " << k << "] residual = " << std::scientific
              << std::sqrt(rsold) << ", ||x|| = " << nx
              << ", ||Ax - b||/||b|| = " << res << std::endl;
  #endif
}
*/

void CGSolver::solve(std::vector<double> & x) {
  double* d_x = nullptr;
  copy_to_device<double>(d_x, x.data(), m_n);
  
  double* d_r = nullptr;
  zero_malloc_on_device<double>(d_r, m_n);

  double* d_p = nullptr;
  zero_malloc_on_device<double>(d_p, m_n);

  double* d_Ap = nullptr;
  zero_malloc_on_device<double>(d_Ap, m_n);

  double* d_b = nullptr;
  copy_to_device<double>(d_b, m_b.data(), m_n);

  double* d_A = nullptr;
  copy_to_device<double>(d_A, m_A.data(), m_m * m_n);

  double* d_tmp = nullptr;
  zero_malloc_on_device<double>(d_tmp, m_n);

  double* d_rsold = nullptr;
  zero_malloc_on_device<double>(d_rsold, 1);

  double* ddot_result = nullptr;
  zero_malloc_on_device<double>(ddot_result, 1);

  double* d_alpha = nullptr;
  zero_malloc_on_device<double>(d_alpha, 1);
  
  double* h_rsnew = nullptr; //used in the for loop below
  
  double* d_beta = nullptr;
  zero_malloc_on_device<double>(d_beta, 1);

  #if DEBUG
    double* h_rsold = nullptr;
  #endif

  int grid_size_m = (m_m + THREADS_PER_BLOCK  - 1) / THREADS_PER_BLOCK;
  // r = b - A * x;
  
  //Might have to switch out with cublas 
  cu_dgemv<double><<<grid_size_m, THREADS_PER_BLOCK>>>(d_Ap, d_A, d_x, 1., 0., m_m, m_n);

  assign_on_device<double>(d_b, d_r, m_n);

  int grid_size_n = (m_n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;  // Round up
  cu_daxpy<double><<<grid_size_n, THREADS_PER_BLOCK>>>(-1.,  d_Ap, d_r, m_n);

  assign_on_device<double>(d_r, d_p, m_n);
  

  cu_ddot<double><<<grid_size_n, THREADS_PER_BLOCK>>>(d_rsold, d_r, d_r, m_n);

  int k = 0;
  for(; k < m_n; k++) {
    // Ap = A * p;
    cu_dgemv<double><<<grid_size_m, THREADS_PER_BLOCK>>>(d_Ap, d_A, d_p, 1., 0., m_m, m_n);
    
    // alpha = rsold / max((p' * Ap), rsold*NEARZERO); (next two lines)
    cu_ddot<double><<<grid_size_n, THREADS_PER_BLOCK>>>(ddot_result, d_p, d_Ap, m_n);

    compute_alpha<double><<<1, 1>>>(d_alpha, d_rsold, ddot_result);

    // x = x + alpha * p;
    cu_daxpy<double><<<grid_size_n, THREADS_PER_BLOCK>>>(d_alpha,  d_p, d_x, m_n);

    // r = r - alpha * Ap; (next two lines)
    cu_negate<double><<<1,1>>>(d_alpha, 1);
    
    cu_daxpy<double><<<grid_size_n, THREADS_PER_BLOCK>>>(d_alpha, d_Ap, d_r, m_n);

    assign_on_device<double>(0., ddot_result);
    cu_ddot<double><<<grid_size_n, THREADS_PER_BLOCK>>>(ddot_result, d_r, d_r, m_n);


    copy_from_device<double>(h_rsnew, ddot_result, 1);

    if(!h_rsnew){
      std::cout << "An error occured copying from device" << std::endl;
      return;
    }

    //At this point h_rsnew is safe to use
    if(*h_rsnew < m_tolerance){
      break;
    }

    //beta = rsnew / rsold;
    compute_beta<double><<<1,1>>>(d_beta, d_rsold, ddot_result); //ddot_result is rsnew 

    //p = r + beta * p;
    assign_on_device(d_r, d_tmp, m_n);
    cu_daxpy<double><<<grid_size_n, THREADS_PER_BLOCK>>>(d_beta, d_p, d_tmp, m_n);
    assign_on_device(d_tmp, d_p, m_n);

    //rsold = rsnew;
    assign_on_device(ddot_result, d_rsold, 1);

    #if DEBUG
      copy_from_device<double>(h_rsold, d_rsold, 1);
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific
                << std::sqrt(*h_rsold) << std::endl;
    #endif
  }

  #if DEBUG
    zero_fill_array<double>(d_r, m_n);
    
    cu_dgemv<double><<<grid_size_n, THREADS_PER_BLOCK>>>(d_r, d_A, d_x, 1., 0., m_m, m_n);

    cu_daxpy<double><<<grid_size_n, THREADS_PER_BLOCK>>>(-1., d_b, d_r, m_n);

    assign_on_device<double>(0., ddot_result);
    cu_ddot<double><<<grid_size_n, THREADS_PER_BLOCK>>>(ddot_result, d_r, d_r, m_n);

    double* h_rsqaured = nullptr;
    copy_from_device<double>(h_rsqaured, ddot_result, 1);

    assign_on_device(0., ddot_result);
    cu_ddot<double><<<grid_size_n, THREADS_PER_BLOCK>>>(ddot_result, d_b, d_b, m_n);
    
    double* h_bsquared = nullptr;
    copy_from_device<double>(h_bsquared, ddot_result, 1);
    double res = std::sqrt(*h_rsqaured) / std::sqrt(*h_bsquared);
    
    assign_on_device(0., ddot_result);
    cu_ddot<double><<<grid_size_n, THREADS_PER_BLOCK>>>(ddot_result, d_x, d_x, m_n);

    double* h_xsquared = nullptr;
    copy_from_device<double>(h_xsquared, ddot_result, 1);
    double nx = std::sqrt(*h_xsquared);

    std::cout << "\t[STEP " << k << "] residual = " << std::scientific
              << std::sqrt(*h_rsold) << ", ||x|| = " << nx
              << ", ||Ax - b||/||b|| = " << res << std::endl;
    
  #endif

  free_on_device<double>(d_x);
  free_on_device<double>(d_r);
  free_on_device<double>(d_p);
  free_on_device<double>(d_Ap);
  free_on_device<double>(d_b);
  free_on_device<double>(d_A);
  free_on_device<double>(d_tmp);
  free_on_device<double>(d_rsold);
  free_on_device<double>(ddot_result);
  free_on_device<double>(d_alpha);
  free_on_device<double>(d_beta);
}




void CGSolver::read_matrix(const std::string & filename) {
  m_A.read(filename);
  m_m = m_A.m();
  m_n = m_A.n();
}


/*
Initialization of the source term b
*/
void Solver::init_source_term(double h) {
  m_b.resize(m_n);

  for (int i = 0; i < m_n; i++) {
    m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) *
             std::sin(10. * M_PI * i * h);
  }
}

