
#include "ceres/marginalizable_manifold.h"

#include "ceres/rotation.h"

namespace ceres {

using RowMat3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

bool SO3Manifold::Plus(const double* x,
                       const double* delta,
                       double* x_plus_delta) const {
  double exp_delta[9];
  AngleAxisToRotationMatrix(delta, RowMajorAdapter3x3(exp_delta));

  auto x_plus_delta_ref = Eigen::Map<RowMat3>(x_plus_delta);
  x_plus_delta_ref =
      Eigen::Map<const RowMat3>(x) * Eigen::Map<const RowMat3>(exp_delta);
  return true;
}

bool SO3Manifold::PlusJacobian(const double* x, double* jacobian) const {
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

bool SO3Manifold::Minus(const double* y,
                        const double* x,
                        double* y_minus_x) const {
  static SO3Minus op;
  op(y, x, y_minus_x);
  return true;
}

// Near y = x, we have
//
// y [-] x = vee(X^T Y - I)
//
// <->
// \delta_1 = 0.5 * ([X^T Y]_{32} - [X^T Y]_{23})
// \delta_2 = 0.5 * ([X^T Y]_{13} - [X^T Y]_{13})
// \delta_3 = 0.5 * ([X^T Y]_{21} - [X^T Y]_{12})
//
// We can differentiate this equation.
bool SO3Manifold::MinusJacobian(const double* x, double* jacobian) const {
  auto J = MatrixRef(jacobian, 3, 9);
  J.setZero();

  auto X = Eigen::Map<const RowMat3>(x);

  for (int i = 0 ; i < 3 ; ++i)
  {
    J(0, i * 3 + 1) = 0.5 * X(i, 2);
    J(1, i * 3 + 2) = 0.5 * X(i, 0);
    J(2, i * 3 + 0) = 0.5 * X(i, 1);
  }

  for (int i = 0 ; i < 3 ; ++i)
  {
    J(0, i * 3 + 2) = -0.5 * X(i, 1);
    J(1, i * 3 + 0) = -0.5 * X(i, 2);
    J(2, i * 3 + 1) = -0.5 * X(i, 0);
  }

  return true;
}

}  // namespace ceres
