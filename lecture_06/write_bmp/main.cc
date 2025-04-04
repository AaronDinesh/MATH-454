/* -------------------------------------------------------------------------- */
#include <cmath>
#include <sstream>
/* -------------------------------------------------------------------------- */
#include "grid.hh"
#include "dumpers.hh"
/* -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- */
static void usage(const std::string & prog_name) {
  std::cerr << prog_name << " <grid_size>" << std::endl;
  exit(0);
}

/* -------------------------------------------------------------------------- */
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int prank, psize;

  MPI_Comm_rank(MPI_COMM_WORLD, &prank);
  MPI_Comm_size(MPI_COMM_WORLD, &psize);

  if (argc != 2) usage(argv[0]);

  std::stringstream args(argv[1]);
  int n;
  args >> n;

  if(args.fail()) usage(argv[0]);

  int m_local = n / psize;

  Grid f(m_local, n);

  float h = 1./n;
  int offset_m = m_local * prank;
  int offset_n = 0;

  for (int i = 0; i < m_local; i++) {
    for (int j = 0; j < n; j++) {
      f(i, j) = std::sin(10. * M_PI * (offset_m + i) * h) *
          std::cos(10. * M_PI * (offset_n + j) * h);
    }
  }

  DumperBinary dumper(f, MPI_COMM_WORLD);
  dumper.dump(0);

  MPI_Finalize();

return 0;
}
