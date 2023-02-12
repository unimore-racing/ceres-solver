#ifndef CERES_INTERNAL_MARGINALIZATION_PRIOR_COST_FUNCTION_H_
#define CERES_INTERNAL_MARGINALIZATION_PRIOR_COST_FUNCTION_H_

#include <Eigen/QR>
#include <iostream>
#include <vector>

#include "ceres/cost_function.h"
#include "ceres/internal/eigen.h"
#include "ceres/marginalizable_manifold.h"

namespace ceres {
namespace internal {

// This class implements the computation of the following cost function
//
//  residual = b + \sum_i A_i * (x_i [-] x_i^0),
//
// where b is an n-dimensional vector, A_i is a matrix of size n x t_i, and x_i
// and x_i^0 are parameters belonging to a manifold with tangent space of size
// t_i, and [-] is the "Minus" function of the manifold for x_i.
class MarginalizationPriorCostFunction : public CostFunction {
 public:
  // The arguments a, reference_points, and manifolds contain A_i, x_i^0, and
  // the manifold for x_i in their i'th element respectively. b contains the
  // vector b.
  MarginalizationPriorCostFunction(
      std::vector<Matrix>&& a,
      Matrix&& b,
      std::vector<Vector>&& reference_points,
      std::vector<const MarginalizableManifold*>&& manifolds)
      : a_(a),
        b_(b),
        reference_points_(reference_points),
        manifolds_(manifolds) {
    // Validate the input
    CHECK_EQ(manifolds_.size(), reference_points_.size());
    CHECK_EQ(a_.size(), reference_points_.size());
    CHECK(!a_.empty());
    for (int i = 0; i < manifolds_.size(); i++) {
      CHECK_EQ(a_[i].cols(), GetTangentSizeForBlock(i));
      CHECK_EQ(a_[i].rows(), b_.rows());
    }

    // Set sizes
    set_num_residuals(b_.size());
    for (const Vector& r : reference_points_) {
      mutable_parameter_block_sizes()->push_back(r.size());
    }

    // Preallocate temporary space for the minus Jacobian.
    int scratch_size_doubles = 0;
    for (const MarginalizableManifold* manifold : manifolds_) {
      // If manifold is nullptr, it is Euclidean. The Euclidean manifold has
      // minus Jacobian identity, which does not require scratch space.
      if (manifold != nullptr) {
        const int minus_jacobian_size =
            manifold->TangentSize() * manifold->AmbientSize();
        scratch_size_doubles =
            std::max(scratch_size_doubles, minus_jacobian_size);
      }
    }
    jacobian_scratch_.resize(scratch_size_doubles);
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const override {
    VectorRef(residuals, num_residuals()) = b_;
    const int num_parameter_blocks = reference_points_.size();

    for (int i = 0; i < num_parameter_blocks; ++i) {
      const int tan_size = GetTangentSizeForBlock(i);

      Vector delta(tan_size);
      if (manifolds_[i] != nullptr) {
        manifolds_[i]->Minus(
            parameters[i], reference_points_[i].data(), delta.data());
      } else {
        // Euclidean manifold
        delta = ConstVectorRef(parameters[i], tan_size) - reference_points_[i];
      }

      VectorRef(residuals, num_residuals()) += a_[i] * delta;
    }

    if (jacobians == nullptr) {
      return true;
    }

    for (int i = 0; i < num_parameter_blocks; ++i) {
      const int ambient_size = reference_points_[i].size();

      if (manifolds_[i] != nullptr) {
        const int tan_size = GetTangentSizeForBlock(i);
        const bool success =
            manifolds_[i]->MinusJacobian2(parameters[i],
                                                reference_points_[i].data(),
                                                jacobian_scratch_.data());
        if (!success)
        {
          return false;
        }
        ConstMatrixRef minus_jacobian(
            jacobian_scratch_.data(), tan_size, ambient_size);
        MatrixRef(jacobians[i], num_residuals(), ambient_size) =
            a_[i] * minus_jacobian;
      } else {
        // Euclidean manifold
        MatrixRef(jacobians[i], num_residuals(), ambient_size) = a_[i];
      }
    }

    return true;
  }

  const std::vector<Matrix>& a() const { return a_; }
  const Vector& b() const { return b_; }

 private:
  int GetTangentSizeForBlock(int block) const {
    if (manifolds_[block] != nullptr) {
      return manifolds_[block]->TangentSize();
    } else {
      // Euclidean manifold. Get the tangent size, also the ambient size, from
      // the reference point.
      return reference_points_[block].rows();
    }
  }

  mutable std::vector<double> jacobian_scratch_;
  std::vector<const MarginalizableManifold*> manifolds_;
  std::vector<Vector> reference_points_;
  Vector b_;
  std::vector<Matrix> a_;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_MARGINALIZATION_PRIOR_COST_FUNCTION_H_
