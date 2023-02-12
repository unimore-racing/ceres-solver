// Author: evanlevine138e@gmail.com (Evan Levine)

#ifndef CERES_INTERNAL_MARGINALIZATION_NUMERICS_H_
#define CERES_INTERNAL_MARGINALIZATION_NUMERICS_H_

#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include "ceres/internal/eigen.h"

namespace ceres {

[[nodiscard]] inline bool TryComputeLDLT(Eigen::LDLT<Matrix>& ldlt,
                                         const Matrix& mat) {
  ldlt.compute(mat);
  if (ldlt.info() != Eigen::Success) {
    return false;
  }
  if ((ldlt.vectorD().array() <= 0.0).any()) {
    return false;
  }

  return true;
}

inline void SymmetrizeWithMean(Matrix& mat) {
  CHECK(mat.rows() == mat.cols());
  for (size_t i = 0; i < mat.rows(); ++i) {
    for (size_t j = i + 1; j < mat.rows(); ++j) {
      const double mean = 0.5 * (mat(j, i) + mat(i, j));
      mat(j, i) = mean;
      mat(i, j) = mean;
    }
  }
}

inline void Symmetrize(Matrix& mat) {
  CHECK(mat.rows() == mat.cols());
  for (size_t i = 0; i < mat.rows(); ++i) {
    for (size_t j = i + 1; j < mat.rows(); ++j) {
      mat(j, i) = mat(i, j);
    }
  }
}

inline void AddScalar(Matrix& mat, double lambda) {
  CHECK(mat.cols() == mat.rows()) << "Matrix must be square!";
  for (size_t i = 0; i < mat.cols(); ++i) {
    mat(i, i) += lambda;
  }
}

template <int p = 2>
[[nodiscard]] inline double GetOperatorNorm(const Matrix& A) {
  if constexpr (p == 2) {
    Eigen::SelfAdjointEigenSolver<Matrix> es(A);
    return es.eigenvalues().lpNorm<Eigen::Infinity>();
  } else {
    static_assert(p == Eigen::Infinity);
    return A.cwiseAbs().rowwise().sum().maxCoeff();
  }
}

[[nodiscard]] inline bool AllEntriesSatisfy(
    const Matrix& mat, std::function<bool(double)> predicate) {
  for (size_t i = 0; i < mat.rows(); ++i) {
    for (size_t j = 0; j < mat.cols(); ++j) {
      if (!predicate(mat(i, j))) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] inline bool IsFinite(const Matrix& mat) {
  return AllEntriesSatisfy(mat,
                           [](double value) { return std::isfinite(value); });
}

[[nodiscard]] inline bool IsSymmetric(const Matrix& A) {
  if (A.rows() != A.cols()) {
    return false;
  }
  for (size_t i = 0; i < A.rows(); ++i) {
    for (size_t j = i + 1; j < A.cols(); ++j) {
      if (A(i, j) != A(j, i)) {
        return false;
      }
    }
  }
  return true;
}

inline void SwapColumns(Matrix& A, size_t i, size_t j) {
  CHECK(i < A.cols());
  CHECK(j < A.cols());
  Vector tmp = A.col(i);
  A.col(i) = A.col(j);
  A.col(j) = tmp;
}

inline void SwapRows(Matrix& A, size_t i, size_t j) {
  CHECK(i < A.rows());
  CHECK(j < A.rows());
  Vector tmp = A.row(i);
  A.row(i) = A.row(j);
  A.row(j) = tmp;
}

class ModifiedLDLT {
 public:
  using MatrixLType =
      const Eigen::TriangularView<const Matrix, Eigen::UnitLower>;
  using MatrixUType =
      const Eigen::TriangularView<const Matrix::AdjointReturnType,
                                  Eigen::UnitUpper>;

  void compute(const Matrix& A, std::optional<double> deltaArg = std::nullopt) {
    CHECK(A.rows() == A.cols());
    const size_t n = A.rows();
    E.resize(n);
    E.setZero();
    D.resize(n);
    D.setZero();

    const double nu = std::max(1.0, sqrt(static_cast<double>(n * n) - 1.0));
    double diag_max = 0.0;
    for (size_t i = 0; i < n; ++i) {
      diag_max = std::max(abs(A(i, i)), diag_max);
    }
    double off_diag_max = 0.0;
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) {
        off_diag_max = std::max(abs(A(i, j)), off_diag_max);
      }
    }
    Delta = deltaArg.value_or(std::numeric_limits<double>::epsilon() *
                              (diag_max + off_diag_max));

    const double betaSq = std::max(
        diag_max,
        std::max(off_diag_max / nu, std::numeric_limits<double>::epsilon()));

    L = A;
    Matrix c(n, n);
    c.setZero();
    for (int i = 0; i < n; ++i) {
      c(i, i) = A(i, i);
    }

    Permutation.resize(n);
    for (int i = 0; i < n; ++i) {
      Permutation[i] = i;
    }

    for (int j = 0; j < n; ++j) {
      int max_idx = j;
      double bestPivotVal = abs(c(j, j));
      for (int i = j + 1; i < n; ++i) {
        const double absVal = abs(c(i, i));
        if (absVal > bestPivotVal) {
          max_idx = i;
          bestPivotVal = absVal;
        }
      }
      if (bestPivotVal == 0.0) {
        IsValid = false;
        return;
      }
      if (max_idx != j) {
        SwapRows(c, j, max_idx);
        SwapColumns(c, j, max_idx);
        SwapRows(L, j, max_idx);
        SwapColumns(L, j, max_idx);
        std::swap(Permutation[j], Permutation[max_idx]);
      }

      // MC4
      for (int s = 0; s < j; ++s) {
        L(j, s) = c(j, s) / D[s];
      }

      // Compute the quantities cij = gij - \sum_s=1^j-1 l_js c_is for i =
      // j+1..n NOTE: this is different from the github page
      //   for (int i = j + 1; i < n; ++i) {
      for (int i = j; i < n; ++i) {
        c(i, j) = L(i, j);
        for (int k = 0; k < j; ++k) {
          c(i, j) -= L(j, k) * c(i, k);
        }
      }

      double theta_j = 0.0;
      if (j < n - 1) {
        theta_j = c.block(j + 1, j, n - j - 1, 1).array().abs().maxCoeff();
      }

      // MC5
      D[j] =
          std::max(std::max(Delta, abs(c(j, j))), theta_j * theta_j / betaSq);

      // MC6
      for (int i = j + 1; i < n; ++i) {
        c(i, i) -= (c(i, j) * c(i, j) / D[j]);
      }
      E[j] = D[j] - c(j, j);
    }

    for (int i = 0; i < n; ++i) {
      L(i, i) = 1.0;
      for (int k = i + 1; k < n; ++k) {
        L(i, k) = 0.0;
      }
    }

    IsValid = IsFinite(D) && IsFinite(L);
  }  // end compute()

  bool is_valid() const { return IsValid; }

  auto matrixPT() const {
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> PT;
    PT.indices() = Permutation;
    return PT;
  }
  const auto& matrixLStorage() const { return L; }
  MatrixLType matrixL() { return MatrixLType(L); }
  Vector VectorD() const { return D; }
  Vector VectorE() const { return E; }
  double delta() const { return Delta; }

  Matrix reconstructedPerturbedMatrix() const {
    auto PT = matrixPT();
    return PT * L * D.asDiagonal() * L.transpose() * PT.transpose();
  }

  double getPerturbationNorm() const
  {
    return E.norm();
  }

 private:
  double Delta;
  bool IsValid = false;
  Matrix L;
  Eigen::Matrix<int, Eigen::Dynamic, 1, Eigen::ColMajor>
      Permutation;  // = P^T in P^T L D L^T P
  Vector D;
  Vector E;
};

}  // namespace ceres

#endif
