// Author: evanlevine138e@gmail.com (Evan Levine)

#ifndef CERES_PUBLIC_MARGINALIZABLE_SO3_MANIFOLD_H_
#define CERES_PUBLIC_MARGINALIZABLE_SO3_MANIFOLD_H_

#include <memory>

#include "ceres/marginalizable_manifold.h"

namespace ceres {

template<typename Scalar>
void QuaternionEigenToCeres(const Scalar* q_eigen, Scalar* q_ceres){
    q_ceres[0] = q_eigen[3];
    q_ceres[1] = q_eigen[0];
    q_ceres[2] = q_eigen[1];
    q_ceres[3] = q_eigen[2];
}

template<typename Scalar>
void QuaternionCeresToEigen(const Scalar* q_ceres, Scalar* q_eigen){
    q_eigen[0] = q_ceres[1];
    q_eigen[1] = q_ceres[2];
    q_eigen[2] = q_ceres[3];
    q_eigen[3] = q_ceres[0];
}

// Functor needed to implement automatically differentiated Minus for
// quaternions.
struct EigenQuaternionMinus {
  template <typename T>
  bool operator()(const T* x_plus_delta, const T* x, T* delta) const {
    // Convert from eigen to ceres format
    T x_plus_delta_ceres[4];
    QuaternionEigenToCeres(x_plus_delta, x_plus_delta_ceres);

    T x_ceres[4];
    QuaternionEigenToCeres(x, x_ceres);

    T x_inverse_ceres[4];
    QuaternionInverse(x_ceres, x_inverse_ceres);

    T x_diff[4];
    QuaternionProduct(x_plus_delta_ceres, x_inverse_ceres, x_diff);

    if (x_diff[0] == T(1)) {
      delta[0] = x_diff[1];
      delta[1] = x_diff[2];
      delta[2] = x_diff[3];
    } else {
      const T cos_sq_delta = x_diff[0] * x_diff[0];
      const T sin_delta = sqrt(T(1.0) - cos_sq_delta);
      const T norm_delta = asin(sin_delta);
      const T delta_by_sin_delta = norm_delta / sin_delta;

      delta[0] = delta_by_sin_delta * x_diff[1];
      delta[1] = delta_by_sin_delta * x_diff[2];
      delta[2] = delta_by_sin_delta * x_diff[3];
    }
    return true;
  }
};

using MarginalizableEigenQuaternionManifold = MarginalizableManifoldAdapterWithAutoDiffMinusJacobian<EigenQuaternionManifold, EigenQuaternionMinus, 4, 3>;

class CERES_EXPORT EigenQuaternionLieGroup
    : public virtual MarginalizableEigenQuaternionManifold,
      public LieGroup {
 public:
  void Between(const double* x,
               const double* y,
               double* xinv_y) const override {
    double x_ceres[4];
    QuaternionEigenToCeres(x, x_ceres);

    double y_ceres[4];
    QuaternionEigenToCeres(y, y_ceres);

    double x_inverse_ceres[4];
    QuaternionInverse(x_ceres, x_inverse_ceres);

    double xinv_y_ceres[4];
    QuaternionProduct(x_inverse_ceres, y_ceres, xinv_y_ceres);

    QuaternionCeresToEigen(xinv_y_ceres, xinv_y);
  }
  // X Y
  void Compose(const double* x, const double* y, double* xy) const override {
    double x_ceres[4];
    QuaternionEigenToCeres(x, x_ceres);
    double y_ceres[4];
    QuaternionEigenToCeres(y, y_ceres);
    double xy_ceres[4];
    QuaternionProduct(x_ceres, y_ceres, xy_ceres);

    QuaternionCeresToEigen(xy_ceres, xy);
  }

  void ComposeJacobian(const double* x,
                       const double* y,
                       double* dxy_dx,
                       double* dxy_dy) const override {
    if (dxy_dx) {
      MatrixRef J(dxy_dx, 4, 4);
      J(0, 0) =  y[3];
      J(0, 1) =  y[2];
      J(0, 2) = -y[1];
      J(0, 3) =  y[0];

      J(1, 0) = -y[2];
      J(1, 1) =  y[3];
      J(1, 2) =  y[0];
      J(1, 3) =  y[1];

      J(2, 0) =  y[1];
      J(2, 1) = -y[0];
      J(2, 2) =  y[3];
      J(2, 3) =  y[2];

      J(3, 0) = -y[0];
      J(3, 1) = -y[1];
      J(3, 2) = -y[2];
      J(3, 3) =  y[3];
    }

    if (dxy_dy) {
      MatrixRef J(dxy_dy, 4, 4);
      J(0, 0) =  x[3];
      J(0, 1) =  -x[2];
      J(0, 2) =  x[1];
      J(0, 3) =  x[0];

      J(1, 0) =  x[2];
      J(1, 1) =  x[3];
      J(1, 2) =  -x[0];
      J(1, 3) =  x[1];

      J(2, 0) =  -x[1];
      J(2, 1) =  x[0];
      J(2, 2) =  x[3];
      J(2, 3) =  x[2];

      J(3, 0) =  -x[0];
      J(3, 1) =  -x[1];
      J(3, 2) =  -x[2];
      J(3, 3) =  x[3];
    }
  }
};


}  // namespace ceres

#endif  // CERES_PUBLIC_MARGINALIZABLE_SO3_MANIFOLD_H_