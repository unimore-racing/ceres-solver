// Author: evanlevine138e@gmail.com (Evan Levine)

// TODO
// Rebase test with Euclidean
// Rebase test with Quaternion in made-up PGO problem?
//

#include "ceres/rebase.h"

#include <bitset>
#include <map>
#include <memory>
#include <numeric>
#include <string>

#include "ceres/cost_function.h"
#include "ceres/covariance.h"
#include "ceres/local_parameterization.h"
#include "ceres/loss_function.h"
#include "ceres/manifold.h"
#include "ceres/parameter_block.h"
#include "ceres/problem.h"
#include "ceres/program.h"
#include "ceres/random.h"
#include "ceres/rebase.h"
#include "ceres/residual_block.h"
#include "ceres/sized_cost_function.h"
#include "ceres/solver.h"
#include "ceres/types.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {
using std::vector;

namespace {
class LinearCostFunction : public CostFunction {
 public:
  LinearCostFunction(vector<Matrix>&& a, const Vector&& b)
      : a_(std::move(a)), b_(b) {
    set_num_residuals(b_.size());
    for (size_t i = 0; i < a_.size(); ++i) {
      mutable_parameter_block_sizes()->push_back(a_[i].cols());
      CHECK(a_[i].rows() == b_.size()) << "Dimensions mismatch";
    }
  }

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const {
    VectorRef res(residuals, b_.rows());
    res = b_;
    for (size_t i = 0; i < a_.size(); ++i) {
      ConstVectorRef pi(parameters[i], parameter_block_sizes()[i]);
      res += a_[i] * pi;
    }

    if (jacobians) {
      for (size_t i = 0; i < a_.size(); ++i) {
        if (jacobians[i]) {
          MatrixRef(jacobians[i], a_[i].rows(), a_[i].cols()) = a_[i];
        }
      }
    }
    return true;
  }

 private:
  const vector<Matrix> a_;
  const Vector b_;
};

LinearCostFunction* MakeRandomLinearCostFunction(
    int num_residuals, const vector<int>& parameter_block_sizes) {
  vector<Matrix> a;
  a.reserve(parameter_block_sizes.size());
  for (int size : parameter_block_sizes) {
    a.push_back(Matrix::Random(num_residuals, size));
  }
  return new LinearCostFunction(std::move(a), Vector::Random(num_residuals));
}

class TestGraphState {
  // This class represents the graph below and facilitates various operations
  // for testing.
  //
  //  +--x0-----x1----prior
  //  |  |     / |
  //  |  |    /  |
  //  |  |   /   |
  //  |  |  /    |
  //  |  | /     |
  //  |  x2     x3
  //  |   \    /
  //  |    \  /
  //  |     \/
  //  +-----x4-------x5
  //
  // Cost functions are random linear cost functions.
 public:
  static constexpr int kNumBlocks = 6;
  static constexpr int kAmbientSize = 4;
  // using ManifoldType = EuclideanLieGroup<kAmbientSize>;
  using ManifoldType = QuaternionLieGroup;
  TestGraphState() {
    Problem::Options options;
    options.cost_function_ownership = DO_NOT_TAKE_OWNERSHIP;
    options.loss_function_ownership = DO_NOT_TAKE_OWNERSHIP; // TODO check this
    problem_ = Problem(options);

    parameter_sizes_.resize(kNumBlocks);
    std::fill(parameter_sizes_.begin(), parameter_sizes_.end(), kAmbientSize);

    // Calculate block offsets
    cum_parameter_sizes_.resize(parameter_sizes_.size(), 0);
    std::partial_sum(parameter_sizes_.begin(),
                     parameter_sizes_.end() - 1,
                     cum_parameter_sizes_.begin() + 1,
                     std::plus<int>());

    // Set the state vec
    state_vector_.resize(kNumBlocks * kAmbientSize);
    for (int k = 0; k < kNumBlocks; ++k) {
      VectorRef pb_vec(&state_vector_(cum_parameter_sizes_[k]), kAmbientSize);
      pb_vec = Vector::Random(kAmbientSize);
      pb_vec /= pb_vec.norm();
    }

    // Get block pointers.
    for (int k = 0; k < kNumBlocks; ++k) {
      double* parameter_block = &state_vector_(cum_parameter_sizes_[k]);
      ordered_parameter_blocks_.push_back(parameter_block);
      parameter_block_names_[parameter_block] = "x" + std::to_string(k);
    }

    // Add parameters.
    for (int k = 0; k < kNumBlocks; ++k) {
      problem_.AddParameterBlock(ordered_parameter_blocks_[k], kAmbientSize);
      problem_.SetManifold(ordered_parameter_blocks_[k], new ManifoldType());
    }

    // Add residuals.
    auto& unary_cost_function =
        cost_functions_.emplace_back(new LinearCostFunction(
            {Matrix::Random(4, kAmbientSize)}, {Vector::Random(4)}));
    problem_.AddResidualBlock(
        unary_cost_function.get(), nullptr, ordered_parameter_blocks_[1]);
    vector<std::pair<int, int>> edges = {
        {0, 1}, {1, 2}, {2, 0}, {3, 1}, {2, 4}, {3, 4}, {0, 4}, {4, 5}};
    int residuals_for_edge = 3;
    for (const auto [i, j] : edges) {
      const std::vector<int> sizes = {parameter_sizes_[i], parameter_sizes_[j]};
      auto& cost_function = cost_functions_.emplace_back(
          MakeRandomLinearCostFunction(residuals_for_edge++, sizes));
      problem_.AddResidualBlock(cost_function.get(),
                                nullptr,
                                ordered_parameter_blocks_[i],
                                ordered_parameter_blocks_[j]);
    }
  }
  void Perturb(double sigma) {
    vector<double*> parameter_blocks;
    problem_.GetParameterBlocks(&parameter_blocks);
    for (double* pb : parameter_blocks) {
      const int tan_size = problem_.ParameterBlockTangentSize(pb);
      const int size = problem_.ParameterBlockSize(pb);
      const Vector tan_perturbation = sigma * Vector::Random(tan_size);
      // Apply perturbation to this parameter block.
      Vector pb_perturbed(size);
      manifold_.Plus(pb, tan_perturbation.data(), pb_perturbed.data());
      VectorRef(pb, size) = pb_perturbed;
    }
  }

  double* GetParameterBlockPtr(int idx) {
    return &state_vector_[cum_parameter_sizes_[idx]];
  }

  const double* GetParameterBlockPtr(int idx) const {
    return const_cast<TestGraphState*>(this)->GetParameterBlockPtr(idx);
  }

  Vector GetStateVector() const { return state_vector_; }

  Vector GetStateVectorInWorld(
      const double* pb_base,
      const std::set<const double*>& rebased_parameter_blocks) const {
    Vector state_vector_world = state_vector_;
    for (int b = 0; b < cum_parameter_sizes_.size(); ++b) {
      const double* pb = GetParameterBlockPtr(b);
      if (rebased_parameter_blocks.find((const double*)pb) !=
          rebased_parameter_blocks.end()) {
        double* pb_w = &state_vector_world[cum_parameter_sizes_[b]];
        manifold_.Compose(pb_base, pb, pb_w);
      }
    }
    return state_vector_world;
  }

  void RebaseParameterBlockAndMarkovBlanket(
      int target_idx,
      const double*& pb_base,
      std::set<const double*>& rebased_parameter_blocks,
      std::vector<std::unique_ptr<RebasedCostFunction>>&
          rebased_cost_functions) {
    const double* target_pb = GetParameterBlockPtr(target_idx);
    ceres::RebaseParameterBlockAndMarkovBlanket(parameter_block_names_,
                                                target_pb,
                                                &problem_,
                                                &pb_base,
                                                &rebased_parameter_blocks,
                                                &rebased_cost_functions);
  }

  void SolveProblem() {
    Solver::Options options;
    // options.use_nonmonotonic_steps = true;
    options.min_lm_diagonal = 1e-10;
    options.max_lm_diagonal = options.min_lm_diagonal;
    options.max_num_iterations = 100;
    Solver::Summary summary;
    Solve(options, &problem_, &summary);
  }

  double EvaluateCost() {
    double cost;
    problem_.Evaluate(
        Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
    return cost;
  }

 private:
  std::vector<int> parameter_sizes_;
  Vector state_vector_;
  vector<double*> ordered_parameter_blocks_;
  vector<int> cum_parameter_sizes_;
  Problem problem_;
  ManifoldType manifold_;
  std::vector<std::unique_ptr<CostFunction>> cost_functions_;
  std::map<const double*, std::string> parameter_block_names_;
};

void TestRebase(size_t target_idx) {
  TestGraphState state;
  state.SolveProblem();

  const Vector state_vec_opt = state.GetStateVector();
  const double cost_before = state.EvaluateCost();

  state.Perturb(/*sigma=*/0.2);

  const double* pb_base;
  std::set<const double*> rebased_parameter_blocks;
  std::vector<std::unique_ptr<RebasedCostFunction>> rebased_cost_functions;
  state.RebaseParameterBlockAndMarkovBlanket(
      target_idx, pb_base, rebased_parameter_blocks, rebased_cost_functions);

  state.SolveProblem();

  const Vector state_after_rebase =
      state.GetStateVectorInWorld(pb_base, rebased_parameter_blocks);

  const double state_error =
      (state_vec_opt - state_after_rebase).lpNorm<Eigen::Infinity>();
  EXPECT_LT(state_error, 0.01);

  const double cost_after = state.EvaluateCost();
  const double cost_error = abs(cost_before - cost_after);
  EXPECT_LT(cost_error, 1e-5);
}
}  //  anonymous namespace

TEST(Rebase, Success) {
  // SetRandomState(5);
  for (int idx_base = 0; idx_base < TestGraphState::kNumBlocks; ++idx_base) {
    ASSERT_NO_FATAL_FAILURE(TestRebase(idx_base));
    return;
  }
}


namespace {
std::vector<Matrix> ComputeNumericalJacobians(
    RebasedCostFunction& rebased_cost_f,
    const std::vector<double*>& parameter_blocks) {
  const int ambient_dim = 4;
  static constexpr double kStepSize = 1e-8;
  const int residual_dim = rebased_cost_f.num_residuals();

  // Residuals at the origin
  Vector residuals(residual_dim);
  rebased_cost_f.Evaluate(parameter_blocks.data(), residuals.data(), nullptr);

  // Compute residuals with a step applied.
  std::vector<Matrix> numerical_jacobians(parameter_blocks.size());
  for (size_t i = 0; i < parameter_blocks.size(); ++i) {
    numerical_jacobians[i].resize(residual_dim, ambient_dim);
  }
  for (size_t i = 0; i < parameter_blocks.size(); ++i) {
    for (size_t j = 0; j < ambient_dim; ++j) {
      Vector residuals_plus_step(residual_dim);
      Vector pb_plus_step(ambient_dim);
      for (size_t k = 0; k < ambient_dim; k++) {
        pb_plus_step[k] = parameter_blocks[i][k];
      }
      pb_plus_step[j] += kStepSize;
      pb_plus_step = pb_plus_step;
      std::vector<double*> parameter_blocks_for_eval = parameter_blocks;
      parameter_blocks_for_eval[i] = pb_plus_step.data();

      rebased_cost_f.Evaluate(parameter_blocks_for_eval.data(),
                              residuals_plus_step.data(),
                              nullptr);
      const Vector col = (1 / kStepSize) * (residuals_plus_step - residuals);
      numerical_jacobians[i].col(j) = col;
    }
  }
  return numerical_jacobians;
}
}  // namespace

TEST(Rebase, RebasedCostFunctionJacobianQuaternion) {
  using ManifoldType = QuaternionLieGroup;

  ManifoldType manifold;
  Vector w = Vector::Random(4);
  Vector x = Vector::Random(4);
  Vector y = Vector::Random(4);
  w = w / w.norm();
  x = x / x.norm();
  y = y / y.norm();
  const std::vector<double*> parameter_blocks = {&w[0], &x[0], &y[0]};
  std::map<const double*, std::string> parameter_block_names = {
      {w.data(), "w"}, {x.data(), "x"}, {y.data(), "y"}};

  // Add residuals.
  const size_t residual_dim = 3;

  // Test case where the original cost function already had the base parameter
  // block.
  {
    const int32_t idx_base = 1;
    const std::vector<int> sizes = {4, 4, 4};
    std::unique_ptr<CostFunction> cost_function;
    cost_function.reset(MakeRandomLinearCostFunction(residual_dim, sizes));

    std::vector<Matrix> jacobians(parameter_blocks.size());
    std::vector<double*> jacobian_ptrs;
    for (size_t i = 0; i < parameter_blocks.size(); ++i) {
      jacobians[i].resize(residual_dim, 4);
      jacobian_ptrs.push_back(jacobians[i].data());
    }

    // Evaluate with analytical Jacobians.
    Vector residuals(residual_dim);
    std::vector<bool> parameter_block_is_rebased = {true, false, false};
    RebasedCostFunction rebased_cost_f(*cost_function,
                                       parameter_block_names,
                                       parameter_block_is_rebased,
                                       manifold,
                                       idx_base);
    rebased_cost_f.Evaluate(
        parameter_blocks.data(), residuals.data(), jacobian_ptrs.data());

    const std::vector<Matrix> numerical_jacobians =
        ComputeNumericalJacobians(rebased_cost_f, parameter_blocks);

    for (size_t i = 0; i < numerical_jacobians.size(); ++i) {
      const Matrix err = jacobians[i] - numerical_jacobians[i];
      ASSERT_LT(err.norm(), 1e-6);
    }
  }

  // Test case where the original cost function does not have the base parameter
  // block.
  {
    const std::vector<int> sizes = {4, 4};
    std::unique_ptr<CostFunction> cost_function;
    cost_function.reset(MakeRandomLinearCostFunction(residual_dim, sizes));

    std::vector<Matrix> jacobians(parameter_blocks.size());
    std::vector<double*> jacobian_ptrs;
    for (size_t i = 0; i < parameter_blocks.size(); ++i) {
      jacobians[i].resize(residual_dim, 4);
      jacobian_ptrs.push_back(jacobians[i].data());
    }

    // Evaluate with analytical Jacobians.
    Vector residuals(residual_dim);
    std::vector<bool> parameter_block_is_rebased = {true, false, false};
    RebasedCostFunction rebased_cost_f(*cost_function,
                                       parameter_block_names,
                                       parameter_block_is_rebased,
                                       manifold);
    rebased_cost_f.Evaluate(
        parameter_blocks.data(), residuals.data(), jacobian_ptrs.data());

    const std::vector<Matrix> numerical_jacobians =
        ComputeNumericalJacobians(rebased_cost_f, parameter_blocks);

    for (size_t i = 0; i < numerical_jacobians.size(); ++i) {
      const Matrix err = jacobians[i] - numerical_jacobians[i];
      ASSERT_LT(err.norm(), 1e-6);
    }
  }
}

}  // namespace internal
}  // namespace ceres
