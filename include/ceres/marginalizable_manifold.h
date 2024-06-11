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

template <typename ManifoldType>
class CERES_EXPORT MarginalizableManifoldAdapter
    : public virtual MarginalizableManifold {
 public:
  int AmbientSize() const override { return mManifold.AmbientSize(); }
  int TangentSize() const override { return mManifold.TangentSize(); }
  bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override {
    return mManifold.Plus(x, delta, x_plus_delta);
  }
  bool PlusJacobian(const double* x, double* jacobian) const override {
    return mManifold.PlusJacobian(x, jacobian);
  }
  bool Minus(const double* y,
             const double* x,
             double* y_minus_x) const override {
    return mManifold.Minus(y, x, y_minus_x);
  }
  bool MinusJacobian(const double* x, double* jacobian) const override {
    return mManifold.MinusJacobian(x, jacobian);
  }
  virtual bool MinusJacobian2(const double* x,
                              const double* y,
                              double* jacobian) const = 0;

 private:
  ManifoldType mManifold;
};

template <typename ManifoldType,
          typename MinusFunctor,
          int ManifoldAmbientSize,
          int ManifoldTangentSize>
class CERES_EXPORT MarginalizableManifoldAdapterWithAutoDiffMinusJacobian
    : public virtual MarginalizableManifoldAdapter<ManifoldType> {
 public:
  bool MinusJacobian2(const double* x,
                      const double* y,
                      double* jacobian) const override {
    double delta[ManifoldTangentSize];
    for (int i = 0; i < ManifoldTangentSize; ++i) {
      delta[i] = 0.0;
    }

    const double* parameter_ptrs[2] = {x, y};
    double* jacobian_ptrs[2] = {jacobian, NULL};

    return internal::AutoDifferentiate<
        ManifoldTangentSize,
        internal::StaticParameterDims<ManifoldAmbientSize,
                                      ManifoldAmbientSize>>(
        functor_, parameter_ptrs, ManifoldTangentSize, delta, jacobian_ptrs);
  }

 private:
  MinusFunctor functor_;
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

}  // namespace ceres

#endif  // CERES_PUBLIC_MARGINALIZABLE_MANIFOLD_H_