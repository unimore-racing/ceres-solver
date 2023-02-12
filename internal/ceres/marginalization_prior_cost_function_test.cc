// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: evanlevine138e@gmail.com (Evan Levine)

#include <vector>

#include "ceres/pose2manifold.h"
#include "ceres/marginalizable_manifold.h"
#include "ceres/marginalization_prior_cost_function.h"
#include "ceres/random.h"
#include "ceres/types.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ceres::internal {

TEST(MarginalizationPriorCostFunction, NumericalJacobianTest) {
  // SetRandomState(5);

  static constexpr size_t kNumBlocks = 2;

  auto ambient_sizes = std::array{4, 7};
  auto tan_sizes = std::array{3, 7};
  std::array<std::unique_ptr<MarginalizableManifold>, kNumBlocks> manifolds = {
      std::make_unique<Pose2Manifold>(), nullptr};

  const int residual_dim = 5;

  std::vector<Vector> reference_points(manifolds.size());
  std::vector<const MarginalizableManifold*> manifold_ptrs(manifolds.size());
  std::vector<Matrix> a(manifolds.size());

  for (int i = 0; i < manifolds.size(); ++i) {
    manifold_ptrs[i] = manifolds[i].get();
    if (i == 0) {
      // Quaternion
      Vector rx = Vector::Random(2);
      rx /= rx.norm();
      reference_points[i] = Vector::Random(4);
      reference_points[i].template head<2>() = rx;
    }
    else
    {
      reference_points[i] = Vector::Random(ambient_sizes[i]);
    }

    a[i] = Matrix::Random(residual_dim, tan_sizes[i]);
  }
  const Vector b = Vector::Random(residual_dim);

  // Create random test states by perturbing the reference points.
  std::vector<Vector> states(manifolds.size());
  std::vector<const double*> state_ptrs(manifolds.size());
  Vector residual_expected = b;
  for (int i = 0; i < manifolds.size(); ++i) {
    const Vector delta = Vector::Random(tan_sizes[i]);
    if (manifolds[i]) {
      states[i].resize(ambient_sizes[i]);
      manifolds[i]->Plus(
          reference_points[i].data(), delta.data(), states[i].data());
    } else {
      states[i] = reference_points[i] + delta;
    }
    state_ptrs[i] = states[i].data();
    residual_expected += a[i] * delta;
  }

  // Allocate the Jacobians.
  std::vector<Matrix> jacobians(manifolds.size());
  std::vector<double*> jacobian_ptrs(manifolds.size());
  for (int i = 0; i < manifolds.size(); ++i) {
    jacobians[i].resize(residual_dim, ambient_sizes[i]);
    jacobian_ptrs[i] = jacobians[i].data();
  }

  MarginalizationPriorCostFunction cost_function(
      std::vector<Matrix>(a),
      Vector(b),
      std::vector<Vector>(reference_points),
      std::move(manifold_ptrs));

  // Check the residual
  Vector residual_actual(residual_dim);
  cost_function.Evaluate(
      state_ptrs.data(), residual_actual.data(), jacobian_ptrs.data());
  const double residual_diff =
      (residual_actual - residual_expected).lpNorm<Eigen::Infinity>();
  ASSERT_LT(residual_diff, 1e-14);

  // Compute analytical Jacobians with respect to the tangent space increment by
  // chaining Jacobians.
  std::vector<Matrix> jacobians_wrt_tan_space_increment(
      manifolds.size());  // Stores the Jacobian wrt the tangent
                          // space increment for x_i.
  for (int i = 0; i < manifolds.size(); ++i) {
    if (manifolds[i]) {
      Matrix plus_jacobian(ambient_sizes[i], tan_sizes[i]);
      manifolds[i]->PlusJacobian(states[i].data(), plus_jacobian.data());
      jacobians_wrt_tan_space_increment[i] = jacobians[i] * plus_jacobian;
    } else {
      jacobians_wrt_tan_space_increment[i] = jacobians[i];
    }
  }

  // Compute numerical Jacobians of the residual with respect to the
  const double kStep = 1e-8;
  const double kInv2Step = 1.0 / (2 * kStep);
  std::vector<Vector> states_plus_step(manifolds.size());
  std::vector<double*> states_plus_step_ptrs(manifolds.size());
  for (int i = 0; i < manifolds.size(); i++) {
    states_plus_step[i] = states[i];
    states_plus_step_ptrs[i] = states_plus_step[i].data();
  }

  Vector residual_plus(residual_dim);
  Vector residual_minus(residual_dim);
  for (int i = 0; i < kNumBlocks; ++i) {
    Matrix jacobian_wrt_tan_space_increment_num(residual_dim, tan_sizes[i]);
    Vector step(tan_sizes[i]);
    for (int j = 0; j < tan_sizes[i]; ++j) {
      step.setZero();
      step[j] = kStep;

      // Evaluate the residual with the step.
      if (manifolds[i]) {
        manifolds[i]->Plus(
            states[i].data(), step.data(), states_plus_step[i].data());
      } else {
        states_plus_step[i] = states[i] + step;
      }

      cost_function.Evaluate(
          states_plus_step_ptrs.data(), residual_plus.data(), nullptr);

      // Evaluate the residual with the negated step.
      step[j] = -kStep;

      if (manifolds[i]) {
        manifolds[i]->Plus(
            states[i].data(), step.data(), states_plus_step[i].data());
      } else {
        states_plus_step[i] = states[i] + step;
      }
      cost_function.Evaluate(
          states_plus_step_ptrs.data(), residual_minus.data(), nullptr);

      jacobian_wrt_tan_space_increment_num.col(j) =
          kInv2Step * (residual_plus - residual_minus);
      states_plus_step[i] = states[i];
    }
    const Matrix jacobian_diff = jacobians_wrt_tan_space_increment[i] -
                                 jacobian_wrt_tan_space_increment_num;
    ASSERT_LT(jacobian_diff.lpNorm<Eigen::Infinity>(), 1e-7);
  }
}

}  // namespace ceres::internal
