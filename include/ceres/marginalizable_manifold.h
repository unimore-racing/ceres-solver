// Author: evanlevine138e@gmail.com (Evan Levine)

#ifndef CERES_PUBLIC_MARGINALIZABLE_MANIFOLD_H_
#define CERES_PUBLIC_MARGINALIZABLE_MANIFOLD_H_

#include <memory>

#include "ceres/internal/autodiff.h"
#include "ceres/internal/export.h"
#include "ceres/manifold.h"
#include "ceres/rotation.h"

namespace ceres {

// The class MarginalizableManifold defines the function Minus
// which is needed to compute difference between two parameter blocks
// that could be manifold objects.
class CERES_EXPORT MarginalizableManifold : public virtual Manifold {
 public:
  // Must return a local size x global size matrix J(x,y) satisfying
  //
  // d/ddelta Minus(Plus(x, delta), y) = J(x, y) * d/ddelta Plus(x, delta).
  virtual bool MinusJacobian2(const double* x,
                              const double* y,
                              double* jacobian) const = 0;
};

template <typename Functor, int kAmbientSize, int kTanSize>
class CERES_EXPORT AutoDiffMarginalizableManifold
    : public virtual MarginalizableManifold {
 public:
  AutoDiffMarginalizableManifold() = default;

  // Takes ownership of functor
  explicit AutoDiffMarginalizableManifold(Functor&& functor)
      : functor_(std::move(functor)) {}

  bool MinusJacobian2(const double* x,
                      const double* y,
                      double* jacobian) const override {
    double delta[kTanSize];
    for (int i = 0; i < kTanSize; ++i) {
      delta[i] = 0.0;
    }

    const double* parameter_ptrs[2] = {x, y};
    double* jacobian_ptrs[2] = {jacobian, NULL};

    return internal::AutoDifferentiate<
        kTanSize,
        internal::StaticParameterDims<kAmbientSize, kAmbientSize>>(
        functor_, parameter_ptrs, kTanSize, delta, jacobian_ptrs);
  }

 private:
  Functor functor_;
};

// Functor needed to implement automatically differentiated Minus for
// quaternions.
struct QuaternionMinus {
  template <typename T>
  bool operator()(const T* x_plus_delta, const T* x, T* delta) const {
    T x_inverse[4];
    x_inverse[0] = x[0];
    x_inverse[1] = -x[1];
    x_inverse[2] = -x[2];
    x_inverse[3] = -x[3];

    T x_diff[4];
    QuaternionProduct(x_plus_delta, x_inverse, x_diff);

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

template <int Size>
class CERES_EXPORT MarginalizableEuclideanManifold
    : public virtual EuclideanManifold<Size>,
      public virtual MarginalizableManifold {
 public:
  virtual ~MarginalizableEuclideanManifold() = default;

  bool MinusJacobian2(const double* x,
                      const double* y,
                      double* jacobian_ptr) const override {
    using MatrixJacobian = Eigen::Matrix<double, Size, Size, Eigen::RowMajor>;
    const int size = EuclideanManifold<Size>::AmbientSize();
    Eigen::Map<MatrixJacobian> jacobian(jacobian_ptr, size, size);
    jacobian.setIdentity();
    return true;
  }
};

class CERES_EXPORT LieGroup : public virtual MarginalizableManifold {
 public:
  // inv(x) * y
  virtual void Between(const double* x,
                       const double* y,
                       double* xinv_times_y) const = 0;

  // x o y = S(S^-1(x) * S^-1(y))
  virtual void Compose(const double* x,
                       const double* y,
                       double* x_o_y) const = 0;

  virtual void ComposeJacobian(const double* x,
                               const double* y,
                               double* dxy_dx,
                               double* dxy_dy) const = 0;
};

// Functor needed to implement automatically differentiated Minus for
// SO(3). This is Log(X^T * x_plus_delta).
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

// x is treated as a row-major 3x3 matrix.
class CERES_EXPORT SO3Manifold : public virtual Manifold {
 public:
  int AmbientSize() const override { return 9; }
  int TangentSize() const override { return 3; }

  // x_plus_delta = x * exp(delta)
  bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override;
  bool PlusJacobian(const double* x, double* jacobian) const override;
  bool Minus(const double* y,
             const double* x,
             double* y_minus_x) const override;
  bool MinusJacobian(const double* x, double* jacobian) const override;
};

class CERES_EXPORT MarginalizableSO3Manifold
    : public virtual AutoDiffMarginalizableManifold<SO3Minus, 9, 3>,
      public virtual SO3Manifold {
 public:
  virtual ~MarginalizableSO3Manifold() = default;
};

class CERES_EXPORT MarginalizableQuaternionManifold
    : public virtual AutoDiffMarginalizableManifold<QuaternionMinus, 4, 3>,
      public virtual QuaternionManifold {
 public:
  virtual ~MarginalizableQuaternionManifold() = default;
};

class CERES_EXPORT QuaternionLieGroup
    : public virtual MarginalizableQuaternionManifold,
      public LieGroup {
 public:
  void Between(const double* x,
               const double* y,
               double* xinv_y) const override {
    double x_inverse[4];
    x_inverse[0] = x[0];
    x_inverse[1] = -x[1];
    x_inverse[2] = -x[2];
    x_inverse[3] = -x[3];
    QuaternionProduct(x_inverse, y, xinv_y);
  }
  // X Y
  void Compose(const double* x, const double* y, double* xy) const override {
    QuaternionProduct(x, y, xy);
  }

  void ComposeJacobian(const double* x,
                       const double* y,
                       double* dxy_dx,
                       double* dxy_dy) const override {
    if (dxy_dx) {
      MatrixRef J(dxy_dx, 4, 4);
      J(0, 0) = y[0];
      J(0, 1) = -y[1];
      J(0, 2) = -y[2];
      J(0, 3) = -y[3];
      J(1, 0) = y[1];
      J(1, 1) = y[0];
      J(1, 2) = y[3];
      J(1, 3) = -y[2];
      J(2, 0) = y[2];
      J(2, 1) = -y[3];
      J(2, 2) = y[0];
      J(2, 3) = y[1];
      J(3, 0) = y[3];
      J(3, 1) = y[2];
      J(3, 2) = -y[1];
      J(3, 3) = y[0];
    }
    if (dxy_dy) {
      MatrixRef J(dxy_dy, 4, 4);
      J(0, 0) = x[0];
      J(1, 0) = x[1];
      J(2, 0) = x[2];
      J(3, 0) = x[3];

      J(0, 1) = -x[1];
      J(1, 1) = x[0];
      J(2, 1) = x[3];
      J(3, 1) = -x[2];

      J(0, 2) = -x[2];
      J(1, 2) = -x[3];
      J(2, 2) = x[0];
      J(3, 2) = x[1];

      J(0, 3) = -x[3];
      J(1, 3) = x[2];
      J(2, 3) = -x[1];
      J(3, 3) = x[0];
    }
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
    const int size = EuclideanManifold<Size>::AmbientSize();
    for (int i = 0; i < size; ++i) {
      xinv_y[i] = y[i] - x[i];
    }
  }
  // X Y
  void Compose(const double* x, const double* y, double* x_o_y) const override {
    const int size = EuclideanManifold<Size>::AmbientSize();
    for (int i = 0; i < size; ++i) {
      x_o_y[i] = x[i] + y[i];
    }
  }

  void ComposeJacobian(const double* x,
                       const double* y,
                       double* dxy_dx,
                       double* dxy_dy) const override {
    const int size = EuclideanManifold<Size>::AmbientSize();
    if (dxy_dx) {
      MatrixRef J(dxy_dx, size, size);
      J.setIdentity();
    }
    if (dxy_dy) {
      MatrixRef J(dxy_dy, size, size);
      J.setIdentity();
    }
  }
};

}  // namespace ceres

#endif  // CERES_PUBLIC_MARGINALIZABLE_MANIFOLD_H_