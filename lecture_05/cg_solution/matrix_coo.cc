#include "matrix_coo.hh"
extern "C" {
#include "mmio.h"
}
#include <mpi.h>

void MatrixCOO::read(const std::string & fn) {
  int nz;

  int ret_code;
  MM_typecode matcode;
  FILE * f;

  if ((f = fopen(fn.c_str(), "r")) == NULL) {
    printf("Could not open matrix");
    exit(1);
  }

  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    exit(1);
  }

  // Matrix is sparse
  if (not(mm_is_matrix(matcode) and mm_is_coordinate(matcode))) {
    printf("Sorry, this application does not support ");
    printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    exit(1);
  }

  if ((ret_code = mm_read_mtx_crd_size(f, &m_m, &m_n, &nz)) != 0) {
    exit(1);
  }

  /* Splitting matrix per rows. */
  int prank, psize;
  MPI_Comm_rank(MPI_COMM_WORLD, &prank);
  MPI_Comm_size(MPI_COMM_WORLD, &psize);

  const int n_rows_local = m_m / psize;
  m_m_start = n_rows_local * prank;
  m_m_end = n_rows_local * (prank + 1);

  /* Memory reservation to avoid repeated
     memory allocations, but not required */
  irn.reserve(nz / psize);
  jcn.reserve(nz / psize);
  a.reserve(nz / psize);

  /*  NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
  m_is_sym = mm_is_symmetric(matcode);
  for (int i = 0; i < nz; i++) {
    int I, J;
    double a_;

    fscanf(f, "%d %d %lg\n", &I, &J, &a_);
    I--; /* adjust from 1-based to 0-based */
    J--;

    if ((m_m_start <= I && I < m_m_end) ||
        (m_is_sym && m_m_start <= J && J < m_m_end)) {
      irn.push_back(I);
      jcn.push_back(J);
      a.push_back(a_);
    }
  }

  if (f != stdin) {
    fclose(f);
  }

}
