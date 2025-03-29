#include <algorithm>
#include <string>
#include <vector>

#ifndef __MATRIX_COO_H_
#define __MATRIX_COO_H_

class MatrixCOO {
public:
  MatrixCOO() = default;

  inline int m() const { return m_m; }
  inline int n() const { return m_n; }

  inline int nz() const { return irn.size(); }
  inline int is_sym() const { return m_is_sym; }

  void read(const std::string & filename);

  //Takes in a vector x and multiplies it with the matrixCOO A and assigns it to result
  void mat_vec(const std::vector<double>& x, std::vector<double>& result);
  void read_distributed(const std::string& filename, MPI_Comm comm);

  //Contains the row indices
  std::vector<int> irn;
  //Contains the column indices
  std::vector<int> jcn;
  //Contains the actual values
  std::vector<double> a;

private:
  int m_m{0};
  int m_n{0};
  bool m_is_sym{false};
};

#endif // __MATRIX_COO_H_
