#ifndef CERES_PUBLIC_POSE2_MANIFOLD_H_
#define CERES_PUBLIC_POSE2_MANIFOLD_H_

#include <memory>

#include "ceres/internal/autodiff.h"
#include "ceres/internal/export.h"
#include "ceres/liegroups/se2.h"
#include "ceres/manifold.h"
#include "ceres/marginalizable_manifold.h"
#include "ceres/rotation.h"
#include "so2manifold.h"

namespace ceres {

struct MyPose2dBase {
  double rp[4];

  MyPose2dBase() = default;

  MyPose2dBase(const double* x) {
    for (int i = 0; i < 4; ++i) {
      rp[i] = x[i];
    }
  }

  double* r() { return &rp[0]; }
  double* p() { return &rp[2]; }

  const double* r() const { return &rp[0]; }
  const double* p() const { return &rp[2]; }

  double yaw_radians() const { return atan2(r()[1], r()[0]); }

  double x() const { return p()[0]; }
  double y() const { return p()[1]; }
};

class CERES_EXPORT Pose2Manifold : public LieGroup {
 public:
  int AmbientSize() const override { return 4; }
  int TangentSize() const override { return 3; }

  // [Rx * Exp(deltar)]
  // [px + Rx * deltap]
  bool Plus(const double* x_ptr,
            const double* delta,
            double* x_plus_delta) const override {
    // Input variables: [delta trans (2 DoF), delta rot (1 DoF)]
    const double* deltap = &delta[0];
    const double* deltar = &delta[2];

    const MyPose2dBase x(x_ptr);
    const double* px = x.p();

    //  Output variables
    double* x_plus_delta_r = &x_plus_delta[0];
    double* x_plus_delta_p = &x_plus_delta[2];

    SO2Manifold so2man;
    so2man.Plus(x.r(), deltar, x_plus_delta_r);

    double Rdeltap[2];

    liegroups::SO2 Rx(x.r());

    liegroups::transform_point(Rdeltap, Rx, deltap);

    VectorRef(x_plus_delta_p, 2) =
        ConstVectorRef(px, 2) + ConstVectorRef(Rdeltap, 2);

    return true;
  }

  void Between(const double* x_ptr,
               const double* y_ptr,
               double* x_inv_y_ptr) const override {
    const liegroups::SE2<double> x(x_ptr);
    const liegroups::SE2<double> y(y_ptr);
    const liegroups::SE2<double> x_inv_y = inverse(x) * y;
    x_inv_y.ToStorage(x_inv_y_ptr);
  }

  void Compose(const double* x_ptr, const double* y_ptr, double* x_times_y_ptr) const override {
    const liegroups::SE2<double> x(x_ptr);
    const liegroups::SE2<double> y(y_ptr);
    const liegroups::SE2<double> xy = x * y;
    xy.ToStorage(x_times_y_ptr);
  }

  void ComposeJacobian(const double* x,
                       const double* y,
                       double* dxy_dx,
                       double* dxy_dy) const override {
    if (dxy_dx) {
      MatrixRef J(dxy_dx, 4, 4);
      J.setZero();
      // [ a.r[0]  | a.r[1] | a.t[0] | a.t[1] ]
      J(0, 0) = y[0];
      J(0, 1) = -y[1];

      J(1, 0) = y[1];
      J(1, 1) = y[0];

      J(2, 0) = y[2];
      J(2, 1) = y[3];
      J(2, 2) = 1.0;

      J(3, 0) = y[3];
      J(3, 1) = -y[2];
      J(3, 3) = 1.0;
    }
    if (dxy_dy) {
      MatrixRef J(dxy_dy, 4, 4);
      J.setZero();
      // [ y.r[0]  | y.r[1] | y.t[0] | y.t[1] ]
      J(0, 0) = x[0];
      J(0, 1) = -x[1];

      J(1, 0) = x[1];
      J(1, 1) = x[0];

      J(2, 2) = x[0];
      J(2, 3) = x[1];

      J(3, 2) = -x[1];
      J(3, 3) = x[0];
    }
  }

  bool PlusJacobian(const double* x, double* jacobian) const override {
    // Jacobian is 4 x 3
    MatrixRef J(jacobian, AmbientSize(), TangentSize());
    J.setZero();
    J(0, 2) = x[1];
    J(1, 2) = -x[0];

    J(2, 0) = x[0];
    J(2, 1) = x[1];

    J(3, 0) = -x[1];
    J(3, 1) = x[0];

    return true;
  }

  // x (-) y = [ Rx^T (py - px) ] (2 DoF)
  //           [  Log(Rx^T Ry)  ] (1 DoF)
  bool Minus(const double* y_ptr,
             const double* x_ptr,
             double* y_minus_x) const override {
    // Input variables
    const MyPose2dBase x(x_ptr);
    const MyPose2dBase y(y_ptr);
    const double* px = x.p();
    const double* py = y.p();
    const double* rx = x.r();
    const double* ry = y.r();
    // Output variables: [trans (2 DoF), rot (1 DoF)]
    double* y_minus_x_tra = &y_minus_x[0];
    double* y_minus_x_rot = &y_minus_x[2];

    SO2Manifold so2man;
    so2man.Minus(ry, rx, y_minus_x_rot);

    const Vector py_minus_px = ConstVectorRef(py, 2) - ConstVectorRef(px, 2);

    liegroups::SO2 Rx(rx);
    liegroups::transform_point_by_inverse(
        y_minus_x_tra, Rx, py_minus_px.data());

    return true;
  }

  // Must return a local size x global size matrix J(x, y) satisfying
  //
  // d/ddelta Minus(Plus(x, delta), y) = J(x, y) * d/ddelta Plus(x, delta).
  bool ComputeMinusJacobian(const double* x,
                            const double* y,
                            double* jacobian) const override {
    const double* rx = &x[0];
    const double* ry = &y[0];
    const double* px = &x[2];
    const double* py = &y[2];

    // Create B
    Matrix B(3, 3);
    B.setZero();

    const Matrix Rx = liegroups::to_rotation_matrix(rx);
    const Matrix Ry = liegroups::to_rotation_matrix(ry);
    const Matrix RyT_Rx = Ry.transpose() * Rx;

    B(2, 2) = 1.0;
    B.topLeftCorner<2, 2>() = RyT_Rx;

    Matrix A(4, 3);
    PlusJacobian(x, A.data());

    MatrixRef J(jacobian, 3, 4);

    J = B * A.transpose();

    return true;
  }

  bool MinusJacobian(const double* x, double* jacobian) const override {
    MatrixRef J(jacobian, TangentSize(), AmbientSize());
    J.setZero();

    double so2_jacobian[2];
    SO2Manifold so2man;
    so2man.MinusJacobian(x, so2_jacobian);

    J(2, 0) = so2_jacobian[0];
    J(2, 1) = so2_jacobian[1];

    J(0, 2) = x[0];
    J(0, 3) = -x[1];
    J(1, 2) = x[1];
    J(1, 3) = x[0];

    return true;
  }
};

}  // namespace ceres

#endif  // CERES_PUBLIC_POSE2_MANIFOLD_H_