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

  // We partition here the total number of entries (nz) in an
  // even way among all the proccess (except for the last one).

  int prank, psize;
  MPI_Comm_rank(MPI_COMM_WORLD, &prank);
  MPI_Comm_size(MPI_COMM_WORLD, &psize);

  int pnz = nz / psize;
  const int i_start = prank * pnz;
  int i_end = i_start + pnz;

  if (prank == (psize - 1))
  {
    pnz = nz - pnz * (psize - 1);
    i_end = nz;
  }


  /* reserve memory for matrices */
  irn.resize(pnz);
  jcn.resize(pnz);
  a.resize(pnz);

  /*  NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
  m_is_sym = mm_is_symmetric(matcode);
  for (int i = 0; i < i_end; i++) {
    int I, J;
    double a_;

    fscanf(f, "%d %d %lg\n", &I, &J, &a_);
    I--; /* adjust from 1-based to 0-based */
    J--;

    if (i >= i_start)
    {
       irn[i-i_start] = I;
       jcn[i-i_start] = J;
       a[i-i_start] = a_;
    }
  }

  if (f != stdin) {
    fclose(f);
  }
}
