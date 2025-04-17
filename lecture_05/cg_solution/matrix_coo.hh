#include <algorithm>
#include <string>
#include <vector>

#ifndef __MATRIX_COO_H_
#define __MATRIX_COO_H_

class MatrixCOO {
public:
  MatrixCOO() = default;

  inline int m() const { return m_m; }
  inline int m_local() const { return m_m_end - m_m_start; }
  inline int m_start() const { return m_m_start; }
  inline int m_end() const { return m_m_end; }
  inline int n() const { return m_n; }

  inline int nz() const { return irn.size(); }
  inline int is_sym() const { return m_is_sym; }

  void read(const std::string & filename);

  void mat_vec(const std::vector<double> & x, std::vector<double> & y) {
    std::fill_n(y.begin(), y.size(), 0.);

    for (size_t z = 0; z < irn.size(); ++z) {
      auto i = irn[z];
      auto j = jcn[z];
      auto a_ = a[z];

      if (m_m_start <= i && i < m_m_end) {
        y[i-m_m_start] += a_ * x[j];
      }
      if (m_is_sym && (i != j) && m_m_start <= j && j < m_m_end) {
        y[j-m_m_start] += a_ * x[i];
      }
    }
  }

  std::vector<int> irn;
  std::vector<int> jcn;
  std::vector<double> a;

private:
  int m_m{0};
  int m_n{0};
  int m_m_start{0};
  int m_m_end{0};
  bool m_is_sym{false};
};

#endif // __MATRIX_COO_H_
