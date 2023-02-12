// Author: evanlevine138e@gmail.com (Evan Levine)

#include "ceres/marginalization.h"

#include <Eigen/Eigenvalues>
#include <bitset>
#include <memory>
#include <numeric>

#include "ceres/cost_function.h"
#include "ceres/covariance.h"
#include "ceres/loss_function.h"
#include "ceres/manifold.h"
#include "ceres/marginalization_numerics.h"
#include "ceres/parameter_block.h"
#include "ceres/problem.h"
#include "ceres/program.h"
#include "ceres/residual_block.h"
#include "ceres/sized_cost_function.h"
#include "ceres/solver.h"
#include "ceres/types.h"
#include "gtest/gtest.h"
#include "marginalization_prior_cost_function.h"

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

TEST(Marginalization, Simple) {
  // Minimal version of the test "Success" to document usage.
  double x = 4.0;
  double y = 2.0;
  Problem problem;
  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  // (2x + 3y + 4)^2
  problem.AddResidualBlock(
      new LinearCostFunction(
          {2.0 * Matrix::Identity(1, 1), 3.0 * Matrix::Identity(1, 1)},
          4.0 * Vector::Ones(1)),
      nullptr,
      &x,
      &y);
  problem.AddResidualBlock(
      new LinearCostFunction({Matrix::Identity(1, 1)}, 11.0 * Vector::Ones(1)),
      nullptr,
      &x);
  // (2y - 12)^2
  problem.AddResidualBlock(
      new LinearCostFunction({2.0 * Matrix::Identity(1, 1)},
                             -12.0 * Vector::Ones(1)),
      nullptr,
      &y);

  // Achieves an objective value of 0.
  const double xopt = -11.0;
  const double yopt = 6.0;

  // The linearization points can be different from the states.
  double xlp = xopt + 10.0;
  double ylp = yopt - 5.0;
  const std::map<const double*, const double*> linearization_states = {
      {&x, &xlp}, {&y, &ylp}};

  // Marginalize out x
  ASSERT_TRUE(MarginalizeOutVariablesSchur(
      MarginalizationOptionsSchur(),  // Default options
      {&x},                           // blocks to marginalize
      &problem,                       // Problem with x,y to modify
      nullptr,                        // Optional IDs for the prior
      &linearization_states));         // Linearization points

  Solver::Options options;
  options.max_num_iterations = 1;
  options.max_lm_diagonal = options.min_lm_diagonal;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  const double error = y - yopt;
  EXPECT_LT(abs(error), 1e-7f);
}

LinearCostFunction* MakeRandomLinearCostFunction(
    int num_residuals, const vector<int>& parameter_block_sizes) {
  vector<Matrix> a;
  a.reserve(parameter_block_sizes.size());
  for (int size : parameter_block_sizes) {
    a.push_back(Matrix::Random(num_residuals, size));
  }
  return new LinearCostFunction(std::move(a), Vector::Random(num_residuals));
}

// residual = 2/3 * p0 ^ 1.5 - 0.5 * p1^2
class NonlinearBinaryCostFunction : public SizedCostFunction<1, 1, 1> {
 public:
  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const {
    const double p0 = parameters[0][0];
    const double p1 = parameters[1][0];
    residuals[0] = 2.0 / 3.0 * pow(p0, 1.5) - 0.5 * pow(p1, 2.0);
    if (jacobians) {
      if (jacobians[0]) {
        jacobians[0][0] = sqrt(p0);
      }
      if (jacobians[1]) {
        jacobians[1][0] = -p1;
      }
    }
    return true;
  }
};

void TestLinearizationState(const MarginalizationOptionsSchur& options) {
  // States
  const vector<double> final_states = {9.0, 2.0};
  vector<double> states = final_states;
  // States for Jacobian.
  const vector<double> linearization_states = {12.0, 4.0};
  /*
   * This toy problem consists of the residuals
   * r1(x0, x1) = 2/3 * x0^1.5 - 0.5 * x1^2
   * r2(x0) = 2.0 * x0 - 2.0
   *
   * At the linearization state (12,4), the Jacobian for r1 is
   *  [ 12^0.5  -4.0 ]
   *  [ 2.0      0.0 ]
   *
   * The residual is [16 * 3^0.5 - 8, 22]^T
   *
   * One can show that marginalizing out x0 yields the prior
   *
   * prior(x1) = 2 * (x1 - 4.0) + 4 + 3 √3,
   */
  Problem problem;
  problem.AddParameterBlock(&states[0], 1);
  problem.AddParameterBlock(&states[1], 1);
  problem.SetManifold(&states[0], new MarginalizableEuclideanManifold<1>());
  problem.SetManifold(&states[1], new MarginalizableEuclideanManifold<1>());
  problem.AddResidualBlock(
      new NonlinearBinaryCostFunction(), nullptr, &states[0], &states[1]);
  problem.AddResidualBlock(
      new LinearCostFunction({2.0 * Eigen::Matrix<double, 1, 1>::Identity()},
                             Eigen::Matrix<double, 1, 1>(-2.0)),
      nullptr,
      &states[0]);
  const std::map<const double*, const double*>
      parameter_block_linearization_states = {
          {&states[0], &linearization_states[0]},
          {&states[1], &linearization_states[1]}};
  vector<ResidualBlockId> marginalization_prior_ids;
  EXPECT_TRUE(
      MarginalizeOutVariablesSchur(options,
                                   {&states[0]},  // Marginalize the first state
                                   &problem,
                                   &marginalization_prior_ids,
                                   &parameter_block_linearization_states));
  const auto* marginalization_prior =
      static_cast<const MarginalizationPriorCostFunction*>(
          problem.GetCostFunctionForResidualBlock(
              marginalization_prior_ids.front()));
  EXPECT_TRUE(marginalization_prior);
  const vector<Matrix> J = marginalization_prior->a();
  const Vector b = marginalization_prior->b();
  const double b_expected = 4.0 + 3 * sqrt(3.0);
  const double J_expected = 2.0;
  EXPECT_EQ(states[0], final_states[0]);
  EXPECT_EQ(states[1], final_states[1]);
  EXPECT_NEAR(b[0], b_expected, 1e-9);
  EXPECT_NEAR(J[0](0, 0), J_expected, 1e-9);
}
TEST(Marginalization, LinearizationState) {
  MarginalizationOptionsSchur options;
  options.max_ldlt_failure_fix_attempts = 0;
  ASSERT_NO_FATAL_FAILURE(TestLinearizationState(options));
  options.method = MarginalizationOptionsSchur::Method::LDLT;
  ASSERT_NO_FATAL_FAILURE(TestLinearizationState(options));
}

template <typename MarginalizationOptionsType>
class TestGraphState {
  // This class represents the graph below and facilitates various operations
  // for testing, including optimization, marginalization of an arbitrary subset
  // of nodes, perturbation of the states, and covariance calculations.
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
  //  +-----x4
  //
  // Cost functions are random linear cost functions. A combination of Euclidean
  // and subset manifolds are used. Both of these manifolds have plus operators
  // that are linear, ensuring that the marginal is represented exactly.
 public:
  static constexpr int kNumBlocks = 5;
  TestGraphState() {
    // Calculate block offsets
    cum_parameter_sizes_.resize(parameter_sizes_.size(), 0);
    std::partial_sum(parameter_sizes_.begin(),
                     parameter_sizes_.end() - 1,
                     cum_parameter_sizes_.begin() + 1,
                     std::plus<int>());

    // Get block pointers.
    for (int k = 0; k < kNumBlocks; ++k) {
      ordered_parameter_blocks_.push_back(
          &state_vector_(cum_parameter_sizes_[k]));
    }

    // Add parameters.
    for (int k = 0; k < kNumBlocks; ++k) {
      int block_size = parameter_sizes_[k];
      problem_.AddParameterBlock(ordered_parameter_blocks_[k], block_size);
      Manifold* manifold = nullptr;
      // if (k <= 1) {
      // manifold = static_cast<Manifold*>(new SubsetManifold(block_size, {1}));
      // } else if (k <= 3) {
      // manifold = static_cast<Manifold*>(
      // new EuclideanManifold<ceres::DYNAMIC>(block_size));
      // }  // else nullptr (Euclidean manifold)
      problem_.SetManifold(ordered_parameter_blocks_[k], manifold);
    }

    // Add residuals.
    problem_.AddResidualBlock(
        new LinearCostFunction({Matrix::Random(4, 2)}, {Vector::Random(4)}),
        nullptr,
        ordered_parameter_blocks_[1]);
    vector<std::pair<int, int>> edges = {
        {0, 1}, {1, 2}, {2, 0}, {3, 1}, {2, 4}, {3, 4}, {0, 4}};
    int residuals_for_edge = 3;
    for (const auto [i, j] : edges) {
      const std::vector<int> sizes = {parameter_sizes_[i], parameter_sizes_[j]};
      problem_.AddResidualBlock(
          MakeRandomLinearCostFunction(residuals_for_edge++, sizes),
          nullptr,
          ordered_parameter_blocks_[i],
          ordered_parameter_blocks_[j]);
    }
  }
  void Perturb() {
    vector<double*> parameter_blocks;
    problem_.GetParameterBlocks(&parameter_blocks);
    for (int b = 0; b < parameter_blocks.size(); ++b) {
      double* pb = parameter_blocks[b];
      const int tan_size = problem_.ParameterBlockTangentSize(pb);
      const int size = problem_.ParameterBlockSize(pb);
      const Vector tan_perturbation = Vector::Random(tan_size);
      // Apply perturbation to this parameter block.
      const Manifold* manifold = problem_.GetManifold(pb);
      Vector pb_perturbed(size);
      if (manifold) {
        manifold->Plus(pb, tan_perturbation.data(), pb_perturbed.data());
      } else {
        pb_perturbed = VectorRef(pb, size) + tan_perturbation;
      }
      VectorRef(pb, size) = pb_perturbed;
    }
  }
  Vector GetStateVector() const { return state_vector_; }
  Vector GetStateVectorSubset(const std::bitset<kNumBlocks>& selection) const {
    Vector res(state_vector_.size());
    int offset = 0;
    for (int i = 0; i < kNumBlocks; i++) {
      if (selection.test(i)) {
        for (int j = 0; j < parameter_sizes_[i]; j++, offset++) {
          res(offset) = state_vector_(cum_parameter_sizes_[i] + j);
        }
      }
    }
    return res.head(offset);
  }
  void SolveProblem() {
    Solver::Options options;
    options.max_num_iterations = 1;
    options.max_lm_diagonal = options.min_lm_diagonal;
    Solver::Summary summary;
    Solve(options, &problem_, &summary);
  }

  // If linearization_state_vector is not nullptr, use it for linearization.
  void MarginalizeOutVariableSubset(
      const MarginalizationOptionsType& options,
      const Vector* linearization_state_vector,
      const std::bitset<kNumBlocks>& parameter_blocks_to_marginalize_mask) {
    const std::set<double*> parameter_blocks_to_marginalize =
        GetParameterBlockSubset(parameter_blocks_to_marginalize_mask);

    options_ = options;
    std::map<const double*, const double*> parameter_block_linearization_states;
    if (linearization_state_vector) {
      // Get block pointers.
      for (int k = 0; k < kNumBlocks; ++k) {
        parameter_block_linearization_states.emplace(
            ordered_parameter_blocks_[k],
            &linearization_state_vector->operator[](cum_parameter_sizes_[k]));
      }
    }

    if constexpr (std::is_same<MarginalizationOptionsType,
                               MarginalizationOptionsQR>::value) {
      const bool success = MarginalizeOutVariablesQR(
          options_,
          parameter_blocks_to_marginalize,
          &problem_,
          nullptr,
          linearization_state_vector ? &parameter_block_linearization_states
                                     : nullptr);
      EXPECT_TRUE(success);
    } else {
      const bool success = MarginalizeOutVariablesSchur(
          options_,
          parameter_blocks_to_marginalize,
          &problem_,
          nullptr,
          linearization_state_vector ? &parameter_block_linearization_states
                                     : nullptr);
      EXPECT_TRUE(success);
    }
  }
  std::set<double*> GetParameterBlockSubset(
      const std::bitset<kNumBlocks> selection) {
    std::set<double*> subset;
    for (int i = 0; i < ordered_parameter_blocks_.size(); i++) {
      if (selection.test(i)) {
        subset.insert(ordered_parameter_blocks_[i]);
      }
    }
    return subset;
  }
  int GetProblemTangentSize(
      const vector<const double*>& problem_parameter_blocks) {
    return std::accumulate(problem_parameter_blocks.begin(),
                           problem_parameter_blocks.end(),
                           0,
                           [this](int sz, const double* pb) {
                             return sz + problem_.ParameterBlockTangentSize(pb);
                           });
  }
  Matrix GetCovarianceMatrixInTangentSpace(
      vector<const double*>& covariance_blocks) {
    Covariance::Options options;
    Covariance covariance(options);
    const int tan_size = GetProblemTangentSize(covariance_blocks);
    Matrix covariance_matrix(tan_size, tan_size);
    CHECK(covariance.Compute(covariance_blocks, &problem_));
    CHECK(covariance.GetCovarianceMatrixInTangentSpace(
        covariance_blocks, covariance_matrix.data()));
    return covariance_matrix;
  }
  Matrix GetCovarianceMatrixInTangentSpace() {
    vector<double*> problem_parameter_blocks;
    problem_.GetParameterBlocks(&problem_parameter_blocks);
    vector<const double*> covariance_blocks;
    for (const double* pb : problem_parameter_blocks) {
      covariance_blocks.push_back(pb);
    }
    return GetCovarianceMatrixInTangentSpace(covariance_blocks);
  }
  Matrix GetCovarianceMatrixInTangentSpace(
      const std::bitset<kNumBlocks> selection) {
    vector<const double*> covariance_blocks;
    for (int i = 0; i < kNumBlocks; i++) {
      if (selection.test(i)) {
        covariance_blocks.push_back(ordered_parameter_blocks_[i]);
      }
    }
    return GetCovarianceMatrixInTangentSpace(covariance_blocks);
  }

 private:
  const std::array<int, 8> parameter_sizes_ = {2, 2, 1, 1, 2};
  Vector state_vector_ = Vector::Random(8);
  vector<double*> ordered_parameter_blocks_;
  vector<int> cum_parameter_sizes_;
  Problem problem_;
  MarginalizationOptionsType options_;
};

template <typename MarginalizationOptionsType>
void TestMarginalization(
    const MarginalizationOptionsType& options,
    bool use_different_states_for_linearization,
    std::bitset<TestGraphState<MarginalizationOptionsType>::kNumBlocks>
        is_marginalized) {
  TestGraphState<MarginalizationOptionsType> state;
  const Matrix marginal_covariance_expected =
      state.GetCovarianceMatrixInTangentSpace(~is_marginalized);
  state.SolveProblem();
  const Vector state_after_first_solve =
      state.GetStateVectorSubset(~is_marginalized);
  state.Perturb();
  // Also get the perturbed states for linearization.
  const Vector lin_state_vec = state.GetStateVector();
  state.SolveProblem();
  state.Perturb();
  state.MarginalizeOutVariableSubset(
      options,
      use_different_states_for_linearization ? &lin_state_vec : nullptr,
      is_marginalized);
  const Matrix marginal_covariance_actual =
      state.GetCovarianceMatrixInTangentSpace();
  const double cov_error =
      (marginal_covariance_expected - marginal_covariance_actual).norm();
  EXPECT_LT(cov_error, 1e-6);
  // Solve the new problem to compute the marginal mean.
  state.Perturb();
  state.SolveProblem();
  const Vector state_after_marginalization =
      state.GetStateVectorSubset(~is_marginalized);
  const double state_error =
      (state_after_marginalization - state_after_first_solve)
          .lpNorm<Eigen::Infinity>();
  EXPECT_LT(state_error, 1e-6);
}

static void TestSuccess() {
  // SetRandomState(5);
  const size_t kNumTrials = 10;
  for (size_t trial = 0; trial < kNumTrials; ++trial) {
    for (int method = 0; method < 4; ++method) {
      for (int assume_marginalized_block_is_full_rank = 0;
           assume_marginalized_block_is_full_rank < 2;
           ++assume_marginalized_block_is_full_rank) {
        for (int use_different_states_for_linearization = 0;
             use_different_states_for_linearization < 2;
             ++use_different_states_for_linearization) {
          for (unsigned int m = 1;
               m <
               (1u << TestGraphState<MarginalizationOptionsSchur>::kNumBlocks) -
                   1;
               m++) {
            if (method <= 2) {
              MarginalizationOptionsSchur options;
              options.method =
                  static_cast<MarginalizationOptionsSchur::Method>(method);
              options.assume_marginalized_block_is_full_rank =
                  static_cast<bool>(assume_marginalized_block_is_full_rank);
              ASSERT_NO_FATAL_FAILURE(
                  TestMarginalization<MarginalizationOptionsSchur>(
                      options,
                      !static_cast<bool>(
                          use_different_states_for_linearization),
                      m));
            } else {
              MarginalizationOptionsQR options;
              options.rank_threshold = std::numeric_limits<double>::epsilon();
              ASSERT_NO_FATAL_FAILURE(
                  TestMarginalization<MarginalizationOptionsQR>(
                      options,
                      !static_cast<bool>(
                          use_different_states_for_linearization),
                      m));
            }
          }
        }
      }
    }
  }
}
}  //  anonymous namespace

TEST(Marginalization, Success) {
  /*
   * Construct a linear least squares problem with 5 variables. Compute the
   * mean and covariance of the likelihood. Marginalize out a subset of
   * variables, producing a smaller problem. Verify that the marginal
   * mean and covariance computed from this problem matches the corresponding
   * entries in the joint mean and covariance computed previously. Perform this
   * test for all subsets of the variables and with different options.
   */
  ASSERT_NO_FATAL_FAILURE(TestSuccess());
}

TEST(Marginalization, SuccessMarginalizedBlockNotFullRank) {
  for (size_t method = 0; method < 4; ++method) {
    // x[1] has no residuals, so the marginalized block is not full rank.
    double x[3] = {0, 1, 2};
    Problem problem;
    problem.AddParameterBlock(&x[0], 1);
    problem.AddParameterBlock(&x[1], 1);
    problem.AddParameterBlock(&x[2], 1);
    // r1(x1, x2) = x1 - x2 + 1
    problem.AddResidualBlock(
        new LinearCostFunction({Eigen::Matrix<double, 1, 1>::Identity(),
                                -1.0 * Eigen::Matrix<double, 1, 1>::Identity()},
                               Vector::Ones(1)),
        nullptr,
        &x[1],
        &x[2]);
    // r2(x1) = x1 + 1
    problem.AddResidualBlock(
        new LinearCostFunction({Eigen::Matrix<double, 1, 1>::Identity()},
                               Vector::Ones(1)),
        nullptr,
        &x[1]);

    // Marginalizing out x0, x1 yields
    // prior(x2) = 0.5 * sqrt(2) * (x2 - 2) + sqrt(2)

    vector<ResidualBlockId> marginalization_prior_ids;

    if (method <= 2) {
      MarginalizationOptionsSchur options;
      options.method = static_cast<MarginalizationOptionsSchur::Method>(method);
      options.assume_marginalized_block_is_full_rank = false;
      EXPECT_TRUE(MarginalizeOutVariablesSchur(options,
                                               {&x[0], &x[1]},
                                               &problem,
                                               &marginalization_prior_ids,
                                               nullptr));
    } else {
      MarginalizationOptionsQR options;
      EXPECT_TRUE(MarginalizeOutVariablesQR(options,
                                            {&x[0], &x[1]},
                                            &problem,
                                            &marginalization_prior_ids,
                                            nullptr));
    }
    const auto* marginalization_prior =
        static_cast<const MarginalizationPriorCostFunction*>(
            problem.GetCostFunctionForResidualBlock(
                marginalization_prior_ids.front()));
    EXPECT_TRUE(marginalization_prior);
    const vector<Matrix> J = marginalization_prior->a();
    const Vector b = marginalization_prior->b();
    EXPECT_NEAR(J[0](0, 0), sqrt(2.0) / 2.0, 1e-10);
    EXPECT_NEAR(b[0], sqrt(2.0), 1e-10);
  }
}

TEST(Marginalization, FailureNoMarkovBlanket) {
  // No markov blanket
  double x[2] = {0, 0};
  Problem problem;
  problem.AddParameterBlock(x, 2);
  problem.AddResidualBlock(
      new LinearCostFunction({Eigen::Matrix<double, 2, 2>::Identity()},
                             Eigen::Vector2d(1.0, 2.0)),
      nullptr,
      x);
  MarginalizationOptionsSchur options;
  EXPECT_FALSE(MarginalizeOutVariablesSchur(options,
                                            {x},  // marginalize x
                                            &problem,
                                            nullptr));
}

TEST(Marginalization, SO3ManifoldTest) {}

TEST(Marginalization, FailureIncorrectlyAssumeFullRank) {
  // Assumed full rank, but system is not full rank.
  double x[2] = {1, 2};
  double y[2] = {3, 4};
  Problem problem;
  problem.AddParameterBlock(x, 2);
  problem.AddParameterBlock(y, 2);
  std::vector<Matrix> jacobians = {Matrix::Zero(3, 2), Matrix::Zero(3, 2)};
  problem.AddResidualBlock(
      new LinearCostFunction(std::move(jacobians), Vector::Random(3)),
      nullptr,
      x,
      y);
  MarginalizationOptionsSchur options;

  options.method = MarginalizationOptionsSchur::Method::LDLT;
  options.max_ldlt_failure_fix_attempts = 0;
  EXPECT_FALSE(
      MarginalizeOutVariablesSchur(options,
                                   {x},  // marginalize x
                                   &problem,
                                   /* marginalization_prior_ids = */ nullptr));
}

TEST(Marginalization, FailureRankZero) {
  for (int method = 0; method < 4; ++method) {
    // Do not assume full rank, but rank 0.
    double x[2] = {1, 2};
    double y[2] = {3, 4};
    Problem problem;
    problem.AddParameterBlock(x, 2);
    problem.AddParameterBlock(y, 2);
    problem.AddResidualBlock(
        new LinearCostFunction({Matrix::Zero(3, 2), Matrix::Zero(3, 2)},
                               Vector::Random(3)),
        nullptr,
        x,
        y);
    if (method <= 2) {
      MarginalizationOptionsSchur options;
      options.method = static_cast<MarginalizationOptionsSchur::Method>(method);
      options.max_ldlt_failure_fix_attempts = 0;
      EXPECT_FALSE(MarginalizeOutVariablesSchur(
          options,
          {x},  // marginalize x
          &problem,
          /* marginalization_prior_ids = */ nullptr));
    } else {
      MarginalizationOptionsQR options;
      EXPECT_FALSE(
          MarginalizeOutVariablesQR(options,
                                    {x},  // marginalize x
                                    &problem,
                                    /* marginalization_prior_ids = */ nullptr));
    }
  }
}

TEST(Marginalization, FailureMarginalizedBlockIsInfeasible) {
  for (int method = 0; method < 4; ++method) {
    double x = 4.0;
    double y = 2.0;
    Problem problem;
    problem.AddParameterBlock(&x, 1);
    problem.AddParameterBlock(&y, 1);
    // Bounds constraints are allowed only for the Markov blanket variables.
    problem.SetParameterUpperBound(&y, 0, 3.0);
    problem.SetParameterLowerBound(&y, 0, 1.0);
    problem.AddResidualBlock(
        new NonlinearBinaryCostFunction(), nullptr, &x, &y);
    // Markov blanket is infeasible.
    y = 4.0;
    if (method <= 2) {
      MarginalizationOptionsSchur options;
      options.method = static_cast<MarginalizationOptionsSchur::Method>(method);
      options.max_ldlt_failure_fix_attempts = 0;
      EXPECT_FALSE(MarginalizeOutVariablesSchur(
          options,
          {&x},  // marginalize x
          &problem,
          /* marginalization_prior_ids = */ nullptr));
    } else {
      MarginalizationOptionsQR options;
      EXPECT_FALSE(
          MarginalizeOutVariablesQR(options,
                                    {&x},  // marginalize x
                                    &problem,
                                    /* marginalization_prior_ids = */ nullptr));
    }
  }
}

TEST(Marginalization, FailureBlockNotInProblem) {
  double x = 4.0;
  double y = 2.0;
  Problem problem;
  problem.AddParameterBlock(&x, 1);
  MarginalizationOptionsSchur options;
  EXPECT_DEATH_IF_SUPPORTED(
      MarginalizeOutVariablesSchur(options,
                                   {&y},  // marginalize y
                                   &problem,
                                   /* marginalization_prior_ids = */ nullptr),
      "Parameter block to marginalize is not in the problem. Did you forget to "
      "add it?");
}

TEST(Marginalization, FailureOnlySomeLinearizationStatesProvided) {
  // If linearization states are provided, they must include all variables.
  double x = 4.0;
  double x_linearization_state = 4.2;
  double y = 2.0;
  Problem problem;
  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddResidualBlock(new NonlinearBinaryCostFunction(), nullptr, &x, &y);

  MarginalizationOptionsSchur options;
  const std::map<const double*, const double*>
      parameter_block_linearization_states = {{&x, &x_linearization_state}};

  EXPECT_DEATH_IF_SUPPORTED(
      MarginalizeOutVariablesSchur(options,
                                   {&x},  // marginalize x
                                   &problem,
                                   /* marginalization_prior_ids = */ nullptr,
                                   &parameter_block_linearization_states),
      "If linearization states are provided, all should be provided");
}

TEST(Marginalization, FailureVariableToMarginalizeHasBoundsConstraints) {
  double x = 4.0;
  double y = 2.0;
  Problem problem;
  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.SetParameterUpperBound(&x, 0, 3.0);
  problem.AddResidualBlock(new NonlinearBinaryCostFunction(), nullptr, &x, &y);
  MarginalizationOptionsSchur options;
  // Variable to marginalize is infeasible.
  EXPECT_FALSE(
      MarginalizeOutVariablesSchur(options,
                                   {&x},  // marginalize x
                                   &problem,
                                   /* marginalization_prior_ids = */ nullptr));
}

TEST(LinearCostFunction, JacobianTest) {
  const int num_residuals = 4;
  const vector<int> parameter_block_sizes = {1, 2, 3};

  const int state_dim = std::accumulate(
      parameter_block_sizes.begin(), parameter_block_sizes.end(), 0);
  const int num_parameter_blocks = parameter_block_sizes.size();

  vector<Matrix> jacobians_expected;
  jacobians_expected.reserve(parameter_block_sizes.size());
  for (int size : parameter_block_sizes) {
    jacobians_expected.push_back(Matrix::Random(num_residuals, size));
  }

  Vector b = Vector::Random(num_residuals);
  Vector x = Vector::Random(state_dim);
  const vector<const double*> parameters = {&x(0), &x(1), &x(3)};

  Vector residual_expected = b;
  for (size_t i = 0; i < parameter_block_sizes.size(); ++i) {
    residual_expected +=
        jacobians_expected[i] *
        ConstVectorRef(parameters[i], parameter_block_sizes[i]);
  }

  LinearCostFunction linear_cost_function(
      std::vector<Matrix>(jacobians_expected), std::move(b));
  Vector residual_actual(num_residuals);

  vector<Matrix> jacobians(num_parameter_blocks);
  double* jacobian_ptrs[num_parameter_blocks];

  for (int i = 0; i < num_parameter_blocks; i++) {
    jacobians[i].resize(num_residuals, parameter_block_sizes[i]);
    jacobian_ptrs[i] = jacobians[i].data();
  }

  linear_cost_function.Evaluate(
      parameters.data(), residual_actual.data(), jacobian_ptrs);

  for (size_t i = 0; i < parameters.size(); ++i) {
    EXPECT_DOUBLE_EQ(
        (jacobians[i] - jacobians_expected[i]).lpNorm<Eigen::Infinity>(), 0.0);
  }
  EXPECT_DOUBLE_EQ(
      (residual_actual - residual_expected).lpNorm<Eigen::Infinity>(), 0.0);

  EXPECT_EQ(linear_cost_function.num_residuals(), num_residuals);
  EXPECT_EQ(linear_cost_function.parameter_block_sizes(),
            parameter_block_sizes);
}

TEST(MarginalizationNumeric, RegularModifiedCholesky) {
  // SetRandomState(5);
  constexpr size_t kNumTrials = 1000;
  std::array<double, 2> actualMinEigs = {-1e-4, 1e-4};
  for (const double actualMinEig : actualMinEigs) {
    for (size_t N = 1; N < 15; ++N) {
      for (size_t trial = 0; trial < kNumTrials; ++trial) {
        Matrix R(N, N);

        R = Matrix::Random(N, N);

        Matrix A = R.transpose() * R;
        for (size_t i = 0; i < A.rows(); ++i) {
          for (size_t j = i + 1; j < A.rows(); ++j) {
            A(i, j) = A(j, i);
          }
        }

        Eigen::SelfAdjointEigenSolver<Matrix> es(A);
        const Vector lambdaOrig = es.eigenvalues();
        const double origMinEig = lambdaOrig.minCoeff();
        for (size_t i = 0; i < N; ++i) {
          A(i, i) -= origMinEig;
          A(i, i) += actualMinEig;
        }

        Matrix DeltaA;

        // Modified Cholesky factorization
        ModifiedLDLT ldlt;
        ldlt.compute(A);

        const double delta = ldlt.delta();

        // Compute the ideal perturbation
        {
          Eigen::SelfAdjointEigenSolver<Matrix> es(A);
          const Vector lambda = es.eigenvalues();
          Vector tau(N);
          const Matrix Q = es.eigenvectors();
          for (size_t i = 0; i < N; ++i) {
            if (lambda[i] >= delta) {
              tau[i] = 0.0;
            } else {
              tau[i] = delta - lambda[i];
            }
          }

          DeltaA = Q * tau.asDiagonal() * Q.transpose();
        }
        const Matrix APerturbedIdeal = A + DeltaA;

        ASSERT_TRUE(ldlt.is_valid());

        const Matrix L = ldlt.matrixLStorage();
        const Vector D = ldlt.VectorD();
        Vector sqrtD(D.size());
        for (int i = 0; i < D.size(); ++i) {
          sqrtD[i] = sqrt(D[i]);
        }
        const Matrix APerturbed = ldlt.reconstructedPerturbedMatrix();

        const Matrix dmcPerturbation = APerturbed - A;

        const double minimalPerturbation = GetOperatorNorm<2>(DeltaA);
        const double actualPerturbation = GetOperatorNorm<2>(dmcPerturbation);

        if (actualMinEig > 0.0) {
          ASSERT_LE(actualPerturbation, 1e-2);
        } else {
          ASSERT_LE(actualPerturbation, minimalPerturbation * 75.0);
        }
      }
    }
  }
}

}  // namespace internal
}  // namespace ceres
