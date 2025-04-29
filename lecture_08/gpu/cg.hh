#include "matrix.hh"
#include <string>
#include <vector>

#ifndef __CG_HH__
#define __CG_HH__

/*v
 * oid CGSolver::solve(std::vector<double> & x)
void CGSolver::read_matrix(const std::string & filename)
void CGSolverSparse::solve(std::vector<double> & x)
void CGSolverSparse::read_matrix(const std::string & filename)
void MatrixCSR::mvm(const std::vector<double> & x, std::vector<double> & y)
const void MatrixCSR::loadMMMatrix(const std::string & filename) void
Solver::init_source_term(int n, double h)
*/
class Solver {
public:
  virtual void read_matrix(const std::string & filename) = 0;
  void init_source_term(double h);
  virtual void solve(std::vector<double> & x, size_t THREADS_PER_BLOCK) = 0;

  inline size_t m() const { return (size_t) m_m; }
  inline size_t n() const { return (size_t) m_n; }

  void tolerance(double tolerance) { m_tolerance = tolerance; }

protected:
  int m_m{0};
  int m_n{0};
  std::vector<double> m_b;
  double m_tolerance{1e-10};
};

class CGSolver : public Solver {
public:
  CGSolver() = default;
  virtual void read_matrix(const std::string & filename);
  virtual void solve(std::vector<double> & x, size_t THREADS_PER_BLOCK);
private:
  Matrix m_A;
};

#endif /* __CG_HH__ */
