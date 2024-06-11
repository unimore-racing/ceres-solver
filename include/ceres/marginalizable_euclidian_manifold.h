// Author: evanlevine138e@gmail.com (Evan Levine)

#ifndef CERES_PUBLIC_MARGINALIZABLE_EUCLIDIAN_MANIFOLD_H_
#define CERES_PUBLIC_MARGINALIZABLE_EUCLIDIAN_MANIFOLD_H_

#include <memory>

#include "ceres/marginalizable_manifold.h"

namespace ceres {

template <int Size>
class CERES_EXPORT MarginalizableEuclideanManifold
    : public virtual MarginalizableManifoldAdapter<EuclideanManifold<Size>> {
 public:
  virtual ~MarginalizableEuclideanManifold() = default;

  bool MinusJacobian2(const double* x,
                      const double* y,
                      double* jacobian_ptr) const override {
    using MatrixJacobian = Eigen::Matrix<double, Size, Size, Eigen::RowMajor>;
    Eigen::Map<MatrixJacobian> jacobian(jacobian_ptr, Size, Size);
    jacobian.setIdentity();
    return true;
  }
};

template <int Size>
class CERES_EXPORT EuclideanLieGroup
    : public virtual MarginalizableEuclideanManifold<Size>,
      public LieGroup {
 public:
  void Between(const double* x,
               const double* y,
               double* xinv_y) const override {
    for (int i = 0; i < Size; ++i) {
      xinv_y[i] = y[i] - x[i];
    }
  }

  // X Y
  void Compose(const double* x, const double* y, double* x_o_y) const override {
    for (int i = 0; i < Size; ++i) {
      x_o_y[i] = x[i] + y[i];
    }
  }

  void ComposeJacobian(const double* x,
                       const double* y,
                       double* dxy_dx,
                       double* dxy_dy) const override {
    if (dxy_dx) {
      MatrixRef J(dxy_dx, Size, Size);
      J.setIdentity();
    }

    if (dxy_dy) {
      MatrixRef J(dxy_dy, Size, Size);
      J.setIdentity();
    }
  }
};

}  // namespace ceres

#endif  // CERES_PUBLIC_MARGINALIZABLE_MANIFOLD_H_