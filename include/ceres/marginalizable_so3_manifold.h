// Author: evanlevine138e@gmail.com (Evan Levine)

#ifndef CERES_PUBLIC_MARGINALIZABLE_EIGEN_QUATERNION_MANIFOLD_H_
#define CERES_PUBLIC_MARGINALIZABLE_EIGEN_QUATERNION_MANIFOLD_H_

#include <memory>

#include "ceres/marginalizable_manifold.h"

namespace ceres {

// Functor needed to implement automatically differentiated Minus for SO(3).
// This is Log(X^T * x_plus_delta).
struct SO3Minus {
  template <typename T>
  bool operator()(const T* x_plus_delta, const T* x, T* delta) const {
    const auto X = RowMajorAdapter3x3(x);
    const auto Y = RowMajorAdapter3x3(x_plus_delta);
    T XTY_storage[9];
    auto XTY = RowMajorAdapter3x3(XTY_storage);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t k = 0; k < 3; ++k) {
        XTY(i, k) = T(0);
        for (size_t j = 0; j < 3; ++j) {
          XTY(i, k) += X(j, i) * Y(j, k);
        }
      }
    }
    RotationMatrixToAngleAxis(
        RowMajorAdapter3x3(static_cast<const T*>(XTY_storage)), delta);
    return true;
  }
};

using RowMat3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

// x is treated as a row-major 3x3 matrix.
class CERES_EXPORT SO3Manifold : public virtual Manifold {
 public:
  int AmbientSize() const override { return 9; }
  int TangentSize() const override { return 3; }

  // x_plus_delta = x * exp(delta)
  bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override {
    double exp_delta[9];
    AngleAxisToRotationMatrix(delta, RowMajorAdapter3x3(exp_delta));

    auto x_plus_delta_ref = Eigen::Map<RowMat3>(x_plus_delta);
    x_plus_delta_ref =
        Eigen::Map<const RowMat3>(x) * Eigen::Map<const RowMat3>(exp_delta);
    return true;
  }

  bool PlusJacobian(const double* x, double* jacobian) const override {
    auto J = MatrixRef(jacobian, 9, 3);
    J.setZero();
    const double R11 = x[0];
    const double R12 = x[1];
    const double R13 = x[2];
    const double R21 = x[3];
    const double R22 = x[4];
    const double R23 = x[5];
    const double R31 = x[6];
    const double R32 = x[7];
    const double R33 = x[8];

    J(0, 2) = R12;
    J(0, 1) = -R13;

    J(3, 2) = R22;
    J(3, 1) = -R23;

    J(6, 2) = R32;
    J(6, 1) = -R33;

    J(1, 2) = -R11;
    J(1, 0) = R13;

    J(4, 2) = -R21;
    J(4, 0) = R23;

    J(7, 2) = -R31;
    J(7, 0) = R33;

    J(2, 1) = R11;
    J(2, 0) = -R12;

    J(5, 1) = R21;
    J(5, 0) = -R22;

    J(8, 1) = R31;
    J(8, 0) = -R32;
    return true;
  }

  bool Minus(const double* y,
             const double* x,
             double* y_minus_x) const override {
    static SO3Minus op;
    op(y, x, y_minus_x);
    return true;
  }

  bool MinusJacobian(const double* x, double* jacobian) const override {
    auto J = MatrixRef(jacobian, 3, 9);
    J.setZero();

    auto X = Eigen::Map<const RowMat3>(x);

    for (int i = 0; i < 3; ++i) {
      J(0, i * 3 + 1) = 0.5 * X(i, 2);
      J(1, i * 3 + 2) = 0.5 * X(i, 0);
      J(2, i * 3 + 0) = 0.5 * X(i, 1);
    }

    for (int i = 0; i < 3; ++i) {
      J(0, i * 3 + 2) = -0.5 * X(i, 1);
      J(1, i * 3 + 0) = -0.5 * X(i, 2);
      J(2, i * 3 + 1) = -0.5 * X(i, 0);
    }

    return true;
  }
};

using MarginalizableSO3Manifold =
    MarginalizableManifoldAdapterWithAutoDiffMinusJacobian<SO3Manifold,
                                                           SO3Minus,
                                                           9,
                                                           3>;

}  // namespace ceres

#endif  // CERES_PUBLIC_MARGINALIZABLE_MANIFOLD_H_