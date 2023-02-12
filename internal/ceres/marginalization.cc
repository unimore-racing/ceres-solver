// Author: evanlevine138e@gmail.com (Evan Levine)

#include "ceres/marginalization.h"

#include <Eigen/Eigenvalues>
#include <fstream>
#include <memory>

#include "ceres/cost_function.h"
#include "ceres/dense_sparse_matrix.h"
#include "ceres/evaluator.h"
#include "ceres/internal/eigen.h"
#include "ceres/invert_psd_matrix.h"
#include "ceres/loss_function.h"
#include "ceres/manifold.h"
#include "ceres/marginalization_numerics.h"
#include "ceres/marginalization_prior_cost_function.h"
#include "ceres/ordered_groups.h"
#include "ceres/parameter_block.h"
#include "ceres/problem.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/rebase.h"
#include "ceres/reorder_program.h"
#include "ceres/residual_block.h"
#include "ceres/types.h"

namespace ceres {
namespace internal {

using std::map;
using std::set;
using std::vector;

namespace {

// Modified version of ceres::internal::InvertPSDMatrix with checking.
template <int kSize>
[[nodiscard]] bool TryInvertSymmetricPositiveSemidefiniteMatrix(
    bool assume_full_rank,
    const typename EigenTypes<kSize, kSize>::Matrix& m,
    typename EigenTypes<kSize, kSize>::Matrix* mInv) {
  CHECK_NOTNULL(mInv);
  CHECK(m.rows() <= 1 || m(0, 1) == m(1, 0));  // Sanity check symmetry
  using MType = typename EigenTypes<kSize, kSize>::Matrix;
  const int size = m.rows();
  if (assume_full_rank) {
    Eigen::LDLT<Matrix> ldlt(m.rows());
    ldlt.compute(m);

    if (ldlt.info() != Eigen::Success) {
      return false;
    }
    if ((ldlt.vectorD().array() <= 0.0).any()) {
      return false;
    }
    *mInv = ldlt.solve(MType::Identity(size, size));
    return true;
  } else {
    // For a thin SVD the number of columns of the matrix need to be dynamic.
    using SVDMType = typename EigenTypes<kSize, Eigen::Dynamic>::Matrix;
    Eigen::JacobiSVD<SVDMType> svd(m,
                                   Eigen::ComputeThinU | Eigen::ComputeThinV);
    // TODO evlevine c#
    *mInv = svd.solve(MType::Identity(size, size));
    return true;
  }
}

// Compute U,D such that U D U^T = P, where U is n x r, D is r x r, and
// r = rank(P).
//
// Returns "perturbation" containing
[[nodiscard]] bool GetEigendecomposition(
    double rank_threshold,
    const Matrix& P,
    Matrix* U,
    Vector* D,
    std::optional<double>* condition_number,
    double* perturbation) {
  CHECK_NOTNULL(U);
  CHECK_NOTNULL(D);
  CHECK_NOTNULL(condition_number);
  CHECK(P.rows() == P.cols());

  Eigen::SelfAdjointEigenSolver<Matrix> es(P);
  Vector v = es.eigenvalues();
  *U = es.eigenvectors();

  // Compute the rank.
  int rank = 0;
  double perturbationSq = 0.0;
  for (int i = 0; i < v.rows(); ++i) {
    if (v[i] > rank_threshold) {
      ++rank;
    } else {
      perturbationSq += v[i] * v[i];
    }
  }

  if (rank == 0) {
    condition_number->reset();
    return false;
  }

  *D = v.tail(rank);
  const Matrix U2 = U->block(0, U->cols() - rank, U->rows(), rank);
  *U = U2;

  *condition_number = v.array().abs().maxCoeff() / v.array().abs().minCoeff();
  *perturbation = sqrt(perturbationSq);
  return true;
}

[[nodiscard]] int SumTangentSize(
    const vector<ParameterBlock*>& parameter_blocks) {
  int sum = 0;
  for (const ParameterBlock* pb : parameter_blocks) {
    sum += pb->TangentSize();
  }
  return sum;
}

[[nodiscard]] bool HasBoundsConstraintForParameterBlock(const Problem& problem,
                                                        double* pb) {
  const int size = problem.ParameterBlockSize(pb);
  for (int i = 0; i < size; i++) {
    if (problem.GetParameterLowerBound(pb, i) >
        -std::numeric_limits<double>::max()) {
      return true;
    }
    if (problem.GetParameterUpperBound(pb, i) <
        std::numeric_limits<double>::max()) {
      return true;
    }
  }
  return false;
}

// Returns whether there are non-trivial bounds constraints for any parameter
// blocks in a set.
[[nodiscard]] bool HasBoundsConstraintsForAnyParameterBlocks(
    const Problem& problem, const set<double*>& parameter_blocks) {
  for (double* pb : parameter_blocks) {
    if (HasBoundsConstraintForParameterBlock(problem, pb)) {
      return true;
    }
  }
  return false;
}

// Build a problem consisting of the parameter blocks to be marginalized, their
// Markov blanket, and error terms involving the parameter blocks to
// marginalize. The lifetime of external_problem must exceed that of the
// returned problem. If problem_pb_to_storage_map is provided, it is used to
// store copies of the parameter blocks in the new problem, use store pointers
// to the parameter blocks in the original problem in keys. Otherwise, just use
// pointers to parameter blocks in the original problem.
std::unique_ptr<ProblemImpl> BuildProblemWithMarginalizedBlocksAndMarkovBlanket(
    const Problem& external_problem,
    const set<double*>& parameter_blocks_to_marginalize,
    const std::map<const double*, const double*>*
        parameter_block_linearization_states,
    std::map<const double*, double*>* storage_to_problem_pb_map,
    std::map<const double*, Vector>* problem_pb_to_storage_map) {
  // Input validated previously. The external problem contains all parameter
  // blocks to marginalize.
  if (parameter_block_linearization_states) {
    CHECK_NOTNULL(problem_pb_to_storage_map);
    CHECK_NOTNULL(storage_to_problem_pb_map);
    problem_pb_to_storage_map->clear();
    storage_to_problem_pb_map->clear();
  }

  Problem::Options options;
  options.cost_function_ownership = Ownership::DO_NOT_TAKE_OWNERSHIP;
  options.loss_function_ownership = Ownership::DO_NOT_TAKE_OWNERSHIP;
  options.manifold_ownership = Ownership::DO_NOT_TAKE_OWNERSHIP;
  options.enable_fast_removal = true;
  set<ResidualBlockId> marginalized_blocks_residual_ids;

  auto new_problem = std::make_unique<ProblemImpl>(options);

  std::set<double*> blanket_and_marg_blocks = parameter_blocks_to_marginalize;

  auto maybeAddParameterBlockToNewProblem =
      [&](double* pb, int size, bool is_blanket_pb) -> double* {
    double* pb_in_new_problem;
    if (problem_pb_to_storage_map) {
      // If linearization states are different from the current states, we
      // must use new storage for states in the local problem.
      const auto [kv, success] =
          problem_pb_to_storage_map->try_emplace(pb, size);
      CHECK(kv != problem_pb_to_storage_map->end())
          << "Failed to insert into problem_pb_to_storage_map";

      if (success) {
        // We had not allocated storage for this parameter block. Initialize the
        // newly allocated storage.
        const auto kv_lin_state =
            parameter_block_linearization_states->find(pb);
        CHECK(kv_lin_state != parameter_block_linearization_states->end())
            << "If linearization states are provided, all should be "
               "provided";
        const double* lin_state = kv_lin_state->second;

        // Set the state in the local problem to the linearization state.
        Vector& pb_state = kv->second;
        pb_state = ConstVectorRef(lin_state, size);
        pb_in_new_problem = pb_state.data();
        if (is_blanket_pb) {
          // The inverse map will only be used for the Markov blanket parameter
          // blocks.
          CHECK(
              storage_to_problem_pb_map->emplace(pb_in_new_problem, pb).second)
              << "Failed to insert into storage_to_problem_pb_map";
        }
      } else {
        // We have allocated storage for a copy of this parameter block.
        pb_in_new_problem = kv->second.data();
      }
    } else {
      // Re-use storage for the parameter block.
      pb_in_new_problem = pb;
    }

    // Add the parameter block to the new problem if it has not been added
    // already.
    if (new_problem->HasParameterBlock(pb_in_new_problem)) {
      return pb_in_new_problem;
    }
    const Manifold* manifold = external_problem.GetManifold(pb);
    new_problem->AddParameterBlock(
        pb_in_new_problem, size, const_cast<Manifold*>(manifold));
    return pb_in_new_problem;
  };

  // Add the parameter blocks to marginalize to the new problem. Any that have
  // no residuals will not be added in the next step.
  for (double* pb : parameter_blocks_to_marginalize) {
    CHECK(!external_problem.IsParameterBlockConstant(pb))
        << "Constant marginalized blocks are not currently supported in "
           "marginalization";
    const int size = external_problem.ParameterBlockSize(pb);
    maybeAddParameterBlockToNewProblem(pb, size, /*is_blanket_pb=*/false);
  }

  for (double* parameter_block : parameter_blocks_to_marginalize) {
    vector<ResidualBlockId> residual_blocks;
    external_problem.GetResidualBlocksForParameterBlock(parameter_block,
                                                        &residual_blocks);

    for (const ResidualBlockId& residual_block_id : residual_blocks) {
      // Add this residual block if we have not already.
      if (marginalized_blocks_residual_ids.count(residual_block_id)) {
        continue;
      }
      marginalized_blocks_residual_ids.insert(residual_block_id);

      vector<double*> parameter_blocks_for_res;
      external_problem.GetParameterBlocksForResidualBlock(
          residual_block_id, &parameter_blocks_for_res);

      vector<double*> parameter_blocks_in_new_problem;
      parameter_blocks_in_new_problem.reserve(parameter_blocks_for_res.size());

      // Add any new parameter blocks for this residual block.
      for (double* pb_for_res : parameter_blocks_for_res) {
        CHECK(!external_problem.IsParameterBlockConstant(pb_for_res))
            << "Constant parameter blocks are not currently supported in "
               "marginalization";
        const int size = external_problem.ParameterBlockSize(pb_for_res);
        double*& pb_in_new_problem =
            parameter_blocks_in_new_problem.emplace_back();
        const bool is_blanket_pb =
            parameter_blocks_to_marginalize.find(pb_for_res) ==
            parameter_blocks_to_marginalize.end();
        if (is_blanket_pb) {
          blanket_and_marg_blocks.insert(pb_for_res);
        }
        pb_in_new_problem =
            maybeAddParameterBlockToNewProblem(pb_for_res, size, is_blanket_pb);
      }

      const CostFunction* cost_function =
          external_problem.GetCostFunctionForResidualBlock(residual_block_id);
      const LossFunction* loss_function =
          external_problem.GetLossFunctionForResidualBlock(residual_block_id);
      // The new problem is for evaluation. The manifold, loss function, and
      // cost function metadata will not be modified.
      const ResidualBlockId id = new_problem->AddResidualBlock(
          const_cast<CostFunction*>(cost_function),
          const_cast<LossFunction*>(loss_function),
          parameter_blocks_in_new_problem.data(),
          parameter_blocks_in_new_problem.size());
    }

    // Marginalizing a block analytically minimizes it out in the problem. In
    // general, there is no way to guarantee that this minimization always
    // satisfies bounds constraints, so the user should remove bounds
    // constraints on variables to marginalize.
    if (HasBoundsConstraintsForAnyParameterBlocks(external_problem,
                                                  blanket_and_marg_blocks)) {
      return nullptr;
    }
  }
  return new_problem;
}

static constexpr int kMarginalizedGroupId = 0;
static constexpr int kMarkovBlanketGroupId = 1;

// Get an ordering where the parameter blocks to be marginalized are followed by
// parameter blocks in the Markov blanket.
ParameterBlockOrdering GetOrderingForMarginalizedBlocksAndMarkovBlanket(
    const ProblemImpl& problem,
    const set<double*>& parameter_blocks_to_marginalize) {
  ParameterBlockOrdering ordering;
  vector<double*> added_parameter_blocks;
  problem.GetParameterBlocks(&added_parameter_blocks);
  for (double* added_parameter_block : added_parameter_blocks) {
    if (parameter_blocks_to_marginalize.count(added_parameter_block)) {
      ordering.AddElementToGroup(added_parameter_block, kMarginalizedGroupId);
    } else {
      ordering.AddElementToGroup(added_parameter_block, kMarkovBlanketGroupId);
    }
  }
  return ordering;
}

void EvaluateProblem(const int num_elimination_blocks,
                     ProblemImpl* problem,
                     Matrix* jacobian,
                     Vector* residual,
                     Matrix* jtj,
                     Vector* gradient) {
  CHECK_NOTNULL(problem);
  CHECK_NOTNULL(gradient);
  CHECK_NOTNULL(jacobian);

  gradient->resize(problem->program().NumEffectiveParameters());

  std::string error;
  Evaluator::Options evaluator_options;
  // Use DENSE_NORMAL_CHOLESKY, which uses the DenseJacobianWriter required for
  // the cast to DenseSparseMatrix below.
  evaluator_options.linear_solver_type = DENSE_NORMAL_CHOLESKY;
  evaluator_options.num_eliminate_blocks = num_elimination_blocks;
  evaluator_options.context = problem->context();

  std::unique_ptr<Evaluator> evaluator(
      Evaluator::Create(evaluator_options, problem->mutable_program(), &error));
  CHECK(evaluator.get() != nullptr) << "Failed creating evaluator";

  Vector state_vector(problem->NumParameters());
  problem->program().ParameterBlocksToStateVector(state_vector.data());

  double cost;
  std::unique_ptr<SparseMatrix> sparse_jacobian(evaluator->CreateJacobian());
  residual->resize(problem->NumResiduals());
  CHECK(evaluator->Evaluate(state_vector.data(),
                            &cost,
                            residual->data(),
                            gradient->data(),
                            sparse_jacobian.get()));

  // This cast is valid if DenseJacobianWriter is used, ensured by the linear
  // solver choice DENSE_NORMAL_CHOLESKY.
  *jacobian =
      static_cast<const DenseSparseMatrix*>(sparse_jacobian.get())->matrix();
  if (jtj) {
    *jtj = jacobian->transpose() * (*jacobian);
  }
}

// Marginalize out of J x = r
[[nodiscard]] bool MarginalizationHelperSqrtToSqrt(double rank_threshold,
                                                   Matrix& Q2Jp,
                                                   Vector& Q2r,
                                                   size_t tan_size_marginalized,
                                                   Matrix& marg_sqrt_H,
                                                   Vector& marg_sqrt_b) {
  // Implementation is modified from
  // https://github.com/VladyslavUsenko/basalt-mirror

  CHECK(tan_size_marginalized < Q2Jp.cols());
  CHECK(Q2Jp.rows() == Q2r.rows()) << "Q2Jp rows != Q2r.rows()!";

  const size_t keep_size = Q2Jp.cols() - tan_size_marginalized;
  Eigen::Index marg_rank = 0;
  Eigen::Index total_rank = 0;

  {
    const Eigen::Index rows = Q2Jp.rows();
    const Eigen::Index cols = Q2Jp.cols();

    Vector tempVector(cols + 1);
    double* tempData = tempVector.data();

    for (Eigen::Index k = 0; k < cols && total_rank < rows; ++k) {
      Eigen::Index remainingRows = rows - total_rank;
      Eigen::Index remainingCols = cols - k - 1;

      double beta;
      double hCoeff;
      Q2Jp.col(k).tail(remainingRows).makeHouseholderInPlace(hCoeff, beta);

      if (std::abs(beta) > rank_threshold) {
        Q2Jp.coeffRef(total_rank, k) = beta;

        Q2Jp.bottomRightCorner(remainingRows, remainingCols)
            .applyHouseholderOnTheLeft(
                Q2Jp.col(k).tail(remainingRows - 1), hCoeff, tempData + k + 1);
        Q2r.tail(remainingRows)
            .applyHouseholderOnTheLeft(
                Q2Jp.col(k).tail(remainingRows - 1), hCoeff, tempData + cols);
        total_rank++;
      } else {
        Q2Jp.coeffRef(total_rank, k) = 0;
      }

      // Overwrite the householder vectors with 0.
      Q2Jp.col(k).tail(remainingRows - 1).setZero();

      // Save the rank of the marginalized-out part.
      if (k == tan_size_marginalized - 1) {
        marg_rank = total_rank;
      }
    }
  }

  Eigen::Index keep_valid_rows =
      std::max(total_rank - marg_rank, Eigen::Index(1));

  if (total_rank == 0) {
    return false;
  }

  marg_sqrt_H =
      Q2Jp.block(marg_rank, tan_size_marginalized, keep_valid_rows, keep_size);
  marg_sqrt_b = Q2r.segment(marg_rank, keep_valid_rows);

  Q2Jp.resize(0, 0);
  Q2r.resize(0);
  return true;
}

// Compute the Schur complement of the first block of the system jtj, gradient.
// The first block has size block1_size.
[[nodiscard]] bool SchurComplement(const Matrix& jtj,
                                   const Vector& gradient,
                                   bool assume_jtj11_full_rank,
                                   int block1_size,
                                   Matrix* jtj_marginal,
                                   Vector* gradient_marginal) {
  CHECK_NOTNULL(jtj_marginal);
  CHECK_NOTNULL(gradient_marginal);
  CHECK(block1_size < gradient.size());
  CHECK(jtj.rows() == gradient.size() && jtj.rows() == jtj.cols());
  const int tan_size_blanket = gradient.size() - block1_size;
  const Vector gradient1 = gradient.head(block1_size);
  const Vector gradient2 = gradient.tail(tan_size_blanket);
  const Matrix jtj_22 =
      jtj.bottomRightCorner(tan_size_blanket, tan_size_blanket);
  const Matrix jtj_12 = jtj.topRightCorner(block1_size, tan_size_blanket);
  const Matrix jtj_11 = jtj.topLeftCorner(block1_size, block1_size);

  Matrix jtj_11_pinv;
  if (!TryInvertSymmetricPositiveSemidefiniteMatrix<Eigen::Dynamic>(
          assume_jtj11_full_rank, jtj_11, &jtj_11_pinv)) {
    return false;
  }
  *jtj_marginal =
      jtj_22 - jtj_12.transpose() * jtj_11_pinv *
                   jtj_12;  // Naive method of computing a symmetric matrix.
  *gradient_marginal = gradient2 - jtj_12.transpose() * jtj_11_pinv * gradient1;

  // Symmetrize the marginal information from the upper triangle.
  SymmetrizeWithMean(*jtj_marginal);
  return true;
}

// This function computes the cost function for the marginalization prior,
// returning nullptr if unsuccessful. If parameter_block_linearization_states
// are provided, they are used as linearization points and the reference point
// for the marginalization prior cost function returned. The parameter blocks
// for the Markov blanket are turned in blanket_parameter_blocks_problem_states.
template <typename MarginalizationOptionsType, typename SummaryType>
CostFunction* ComputeMarginalizationPrior(
    const MarginalizationOptionsType& options,
    const Problem& external_problem,
    const set<double*>& parameter_blocks_to_marginalize,
    const map<const double*, const double*>*
        parameter_block_linearization_states,
    vector<double*>* blanket_parameter_blocks_problem_states,
    SummaryType* summary) {
  CHECK_NOTNULL(blanket_parameter_blocks_problem_states);

  const bool copy_parameter_blocks =
      parameter_block_linearization_states != nullptr;

  // If linearization states are used, keep copies of the marginalized blocks
  // and their Markov blanket in a map. The keys are the parameter block in the
  // original problem, and the values are the copy.
  std::map<const double*, Vector> problem_pb_to_storage_map;
  // If linearization states are used, the inverse map is used to efficiently
  // recover the Markov blanket parameter blocks in the original problem for use
  // in the marginalization prior cost function.
  std::map<const double*, double*> storage_to_problem_pb_map;
  std::unique_ptr<ProblemImpl> local_problem =
      BuildProblemWithMarginalizedBlocksAndMarkovBlanket(
          external_problem,
          parameter_blocks_to_marginalize,
          parameter_block_linearization_states,
          copy_parameter_blocks ? &storage_to_problem_pb_map : nullptr,
          copy_parameter_blocks ? &problem_pb_to_storage_map : nullptr);

  if (!local_problem) {
    return nullptr;
  }

  set<double*> parameter_blocks_to_marginalize_in_new_prob;
  const set<double*>* marginalized_parameter_blocks_in_loc_prob_ptr = nullptr;
  if (copy_parameter_blocks) {
    for (double* pb : parameter_blocks_to_marginalize) {
      Vector& pb_storage = problem_pb_to_storage_map.at(pb);
      parameter_blocks_to_marginalize_in_new_prob.insert(pb_storage.data());
    }
    marginalized_parameter_blocks_in_loc_prob_ptr =
        &parameter_blocks_to_marginalize_in_new_prob;
  } else {
    marginalized_parameter_blocks_in_loc_prob_ptr =
        &parameter_blocks_to_marginalize;
  }
  const ParameterBlockOrdering ordering =
      GetOrderingForMarginalizedBlocksAndMarkovBlanket(
          *local_problem, *marginalized_parameter_blocks_in_loc_prob_ptr);
  std::string error;
  CHECK(ApplyOrdering(local_problem->parameter_map(),
                      ordering,
                      local_problem->mutable_program(),
                      &error))
      << "Failed to apply ordering required for marginalization";

  local_problem->mutable_program()->SetParameterOffsetsAndIndex();

  const vector<ParameterBlock*>& ordered_parameter_blocks =
      local_problem->program().parameter_blocks();
  vector<ParameterBlock*> blanket_parameter_blocks;
  vector<ParameterBlock*> ordered_parameter_blocks_marginalized;
  for (int i = 0; i < ordered_parameter_blocks.size(); ++i) {
    ParameterBlock* pb = ordered_parameter_blocks.at(i);
    if (i < ordering.GroupSize(kMarginalizedGroupId)) {
      ordered_parameter_blocks_marginalized.push_back(pb);
    } else {
      blanket_parameter_blocks.push_back(pb);
    }
  }

  if (SumTangentSize(blanket_parameter_blocks) == 0) {
    return nullptr;
  }
  const int num_marginalized_blocks =
      marginalized_parameter_blocks_in_loc_prob_ptr->size();
  const int num_blanket_blocks = blanket_parameter_blocks.size();
  const int tan_size_marginalized =
      SumTangentSize(ordered_parameter_blocks_marginalized);

  if (tan_size_marginalized == 0) {
    return nullptr;
  }

  // Get states and manifolds required for the marginalization prior.
  blanket_parameter_blocks_problem_states->resize(
      num_blanket_blocks);  // in the external problem
  vector<Vector> blanket_reference_points(
      num_blanket_blocks);  // in the new problem
  vector<const MarginalizableManifold*> blanket_manifolds(num_blanket_blocks,
                                                          nullptr);

  for (int i = 0; i < num_blanket_blocks; ++i) {
    const int size = blanket_parameter_blocks.at(i)->Size();
    double* pb_in_new_problem =
        blanket_parameter_blocks.at(i)->mutable_user_state();

    const double* pb_state = blanket_parameter_blocks.at(i)->state();

    double* pb_in_ext_problem =
        copy_parameter_blocks ? storage_to_problem_pb_map.at(pb_in_new_problem)
                              : pb_in_new_problem;
    (*blanket_parameter_blocks_problem_states).at(i) = pb_in_ext_problem;
    blanket_reference_points.at(i) = ConstVectorRef(pb_in_new_problem, size);
    const Manifold* manifold = external_problem.GetManifold(pb_in_ext_problem);
    if (manifold) {
      const MarginalizableManifold* mmanifold =
          dynamic_cast<const MarginalizableManifold*>(manifold);
      CHECK(mmanifold) << "Manifold must derive from MarginalizableManifold.";
      blanket_manifolds.at(i) = mmanifold;
    }
  }

  Matrix jtj;
  Matrix jacobian;
  Vector gradient;
  Vector residual;
  EvaluateProblem(num_marginalized_blocks,
                  local_problem.get(),
                  &jacobian,
                  &residual,
                  &jtj,
                  &gradient);

  Matrix marginalization_prior_A;
  Vector marginalization_prior_b;

  if (summary) {
    if (options.compute_jacobian_condition_number) {
      Eigen::JacobiSVD<Matrix> svd(jacobian);
      summary->condition_number_jacobian =
          svd.singularValues()(0) /
          svd.singularValues()(svd.singularValues().size() - 1);
    } else {
      summary->condition_number_jacobian.reset();
    }
  }

  // QR
  if constexpr (std::is_same<MarginalizationOptionsType,
                             MarginalizationOptionsQR>::value) {
    Matrix jacobian_copy = jacobian;
    Vector residual_copy = residual;
    if (!MarginalizationHelperSqrtToSqrt(options.rank_threshold,
                                         jacobian_copy,
                                         residual_copy,
                                         tan_size_marginalized,
                                         marginalization_prior_A,
                                         marginalization_prior_b)) {
      return nullptr;
    }
  } else {
    Matrix jtj_marginal;
    Vector gradient_marginalization_prior;
    if (!SchurComplement(jtj,
                         gradient,
                         options.assume_marginalized_block_is_full_rank,
                         tan_size_marginalized,
                         &jtj_marginal,
                         &gradient_marginalization_prior)) {
      return nullptr;
    }

    CHECK(IsSymmetric(jtj_marginal)) << "JTJ is not symmetric!\n";

    // Now compute the parameters of the marginalization prior cost function. In
    // the comments below, delta_b = x_b [-] x_b^0, x_b is the concatenation of
    // Markov blanket parameter blocks, x_b^0 is the concatenation of their
    // linearization points, [-] is the product of minus operators for the
    // manifolds in the Markov blanket, and g_t is the gradient for the
    // marginalization prior at the reference point.
    if (options.method == MarginalizationOptionsSchur::Method::ModifiedLDLT) {
      ModifiedLDLT ldltMod;
      ldltMod.compute(jtj_marginal, /*delta=*/std::nullopt);

      if (!ldltMod.is_valid()) {
        return nullptr;
      }

      const Vector D = ldltMod.VectorD();
      Vector sqrtD(D.size());
      Vector invSqrtD(D.size());
      for (int i = 0; i < D.size(); ++i) {
        sqrtD[i] = std::sqrt(D[i]);
        invSqrtD[i] = 1.0 / sqrtD[i];
      }

      marginalization_prior_A = sqrtD.asDiagonal() *
                                ldltMod.matrixLStorage().transpose() *
                                ldltMod.matrixPT().transpose();

      // b = S^{-1} g_t
      //   = (P^T L D^{1/2})^-1 g_t
      //   = D^{-1/2} L^{-1} P g_t
      marginalization_prior_b =
          ldltMod.matrixPT().transpose() * gradient_marginalization_prior;
      ldltMod.matrixL().solveInPlace(marginalization_prior_b);
      marginalization_prior_b = invSqrtD.asDiagonal() * marginalization_prior_b;

      if (summary) {
        summary->ldlt_perturbation_norm = ldltMod.getPerturbationNorm();
      }

    } else if (options.method == MarginalizationOptionsSchur::Method::LDLT) {
      // Compute a Cholesky factorization of full-rank matrix jtj_marginal = L *
      // D L^T. The cost for the marginalization prior is
      //
      // cost = \| A delta_b + (A^{-1})^T g_t \|^2,
      //
      // From LDLT we have A^T A = P^T L D L^T P
      //
      // Let D = SqrtD * SqrtD^T for some SqrtD
      //                  => A^T A = P^T L SqrtD SqrtD^T L^T P
      //                  => A = SqrtD^T L^T P
      //
      // (A^T)^{-1} = (P^T L SqrtD)^{-1}
      //            = SqrtD^{-1} L^{-1} P

      // cost = \| S^T delta_b + S^{-1} g_t \|^2, where S = LD^{1/2}
      const size_t max_ldlt_attempts =
          options.max_ldlt_failure_fix_attempts + 1;
      const double lambda_increase_factor = 2.0;
      const size_t n = jtj_marginal.rows();
      double lambda = 1e-4;
      double totalPerturbation = 0.0;
      size_t ldlt_attempts = 0;
      Eigen::LDLT<Matrix> ldlt(n);
      while (!TryComputeLDLT(ldlt, jtj_marginal)) {
        ++ldlt_attempts;
        // std::cout << "Try to fix LDLT\n";
        if (ldlt_attempts == max_ldlt_attempts) {
          return nullptr;
        }

        AddScalar(jtj_marginal, lambda);
        totalPerturbation += lambda;
        lambda *= lambda_increase_factor;
      }

      const Vector D = ldlt.vectorD();
      Vector D_sqrt(D.size());
      for (size_t i = 0; i < n; ++i) {
        D_sqrt[i] = sqrt(D[i]);
      }

      marginalization_prior_A.resize(n, n);
      marginalization_prior_A.setIdentity();
      marginalization_prior_A =
          ldlt.transpositionsP() * marginalization_prior_A;
      marginalization_prior_A =
          ldlt.matrixU() * marginalization_prior_A;  // U == L^T
      marginalization_prior_A = D_sqrt.asDiagonal() * marginalization_prior_A;

      marginalization_prior_b =
          ldlt.transpositionsP() * gradient_marginalization_prior;
      ldlt.matrixL().solveInPlace(marginalization_prior_b);
      for (size_t i = 0; i < n; ++i) {
        marginalization_prior_b(i) /= D_sqrt(i);
      }

      if (summary) {
        summary->ldlt_perturbation_norm =
            totalPerturbation * jtj_marginal.rows();
      }

    } else {
      CHECK(options.method ==
            MarginalizationOptionsSchur::Method::Eigendecomposition)
          << "Invalid method selected.";
      Matrix U;
      Vector D;
      std::optional<double> condition_num;
      double perturbation;
      // Compute the eigen-decomposition of jtj_marginal
      // jtj_marginal = U * D * U^T, where D contains the nonzero eigenvalues of
      // jtj_marginal on the diagonal. The cost for the marginalization prior is
      //
      // cost = \| D^(1/2) U^T delta_b + D^(-1/2) U^T g_t \|^2,
      //
      // Carlevaris-Bianco, Nicholas, Michael Kaess, and Ryan M. Eustice.
      // "Generic node removal for factor-graph SLAM." IEEE Transactions on
      // Robotics 30.6 (2014): 1371-1385.
      if (!GetEigendecomposition(options.rank_threshold,
                                 jtj_marginal,
                                 &U,
                                 &D,
                                 &condition_num,
                                 &perturbation)) {
        // Rank 0, unexpected.
        return nullptr;
      }

      // std::cout << "cond(jtj_marginal) " << cond(jtj_marginal) << "\n";

      if (summary) {
        summary->condition_number_marginal_information = condition_num;
      }

      if (D.size() == 0) {
        return nullptr;
      }

      const Vector sqrtD = D.array().sqrt();
      marginalization_prior_A = sqrtD.matrix().asDiagonal() * U.transpose();
      marginalization_prior_b = sqrtD.cwiseInverse().matrix().asDiagonal() *
                                U.transpose() * gradient_marginalization_prior;
      if (summary) {
        summary->ldlt_perturbation_norm = perturbation;
      }
    }
  }

  if (!IsFinite(marginalization_prior_A)) {
    return nullptr;
  }
  if (!IsFinite(marginalization_prior_b)) {
    return nullptr;
  }

  // Extract blocks from A for the marginalization prior cost function.
  vector<Matrix> marginalization_prior_A_blocks(
      blanket_reference_points.size());
  const int prior_res_dim = marginalization_prior_A.rows();
  int jacobian_column_offset = 0;
  for (int i = 0; i < blanket_manifolds.size(); i++) {
    int tan_size;
    if (blanket_manifolds.at(i) != nullptr) {
      tan_size = blanket_manifolds.at(i)->TangentSize();
    } else {
      // This is a Euclidean manifold. The reference point's size is the ambient
      // size and also the tangent size, so use this.
      tan_size = blanket_reference_points.at(i).rows();
    }
    marginalization_prior_A_blocks.at(i) = marginalization_prior_A.block(
        0, jacobian_column_offset, prior_res_dim, tan_size);
    jacobian_column_offset += tan_size;
  }

  // std::cout << "Create cost\n";
  return new MarginalizationPriorCostFunction(
      std::move(marginalization_prior_A_blocks),
      std::move(marginalization_prior_b),
      std::move(blanket_reference_points),
      std::move(blanket_manifolds));
}

template <typename MarginalizationOptionsType, typename SummaryType>
bool MarginalizeOutVariables(
    const MarginalizationOptionsType& options,
    const set<double*>& parameter_blocks_to_marginalize,
    Problem* problem,
    vector<ResidualBlockId>* marginalization_prior_ids,
    const map<const double*, const double*>*
        parameter_block_linearization_states,
    SummaryType* summary) {
  {
    std::vector<ResidualBlockId> residual_blocks;
    problem->GetResidualBlocks(&residual_blocks);
    double maxNorm = 0.0;
    for (const ResidualBlockId id : residual_blocks) {
      std::vector<double*> pb;
      problem->GetParameterBlocksForResidualBlock(id, &pb);

      const CostFunction* cf = problem->GetCostFunctionForResidualBlock(id);
      Vector residual(cf->num_residuals());
      cf->Evaluate(pb.data(), residual.data(), nullptr);
      const double rNorm = residual.norm();
      maxNorm = std::max(rNorm, maxNorm);
    }
  }

  vector<double*> blanket_ordered_parameter_blocks;
  CostFunction* new_cost_function =
      ComputeMarginalizationPrior(options,
                                  *problem,
                                  parameter_blocks_to_marginalize,
                                  parameter_block_linearization_states,
                                  &blanket_ordered_parameter_blocks,
                                  summary);

  if (!new_cost_function) {
    return false;
  }

  // Remove marginalized blocks.
  for (double* pb : parameter_blocks_to_marginalize) {
    problem->RemoveParameterBlock(pb);
  }

  // Add the cost function for the marginalization prior to the problem.
  const ResidualBlockId mp_id =
      problem->AddResidualBlock(new_cost_function,
                                nullptr,
                                blanket_ordered_parameter_blocks.data(),
                                blanket_ordered_parameter_blocks.size());
  if (marginalization_prior_ids) {
    *marginalization_prior_ids = {mp_id};
  }

  return true;
}
}  // anonymous namespace

}  // namespace internal

bool MarginalizeOutVariablesSchur(
    const MarginalizationOptionsSchur& options,
    const std::set<double*>& parameter_blocks_to_marginalize,
    Problem* problem,
    std::vector<ResidualBlockId>* marginalization_prior_ids,
    const std::map<const double*, const double*>*
        parameter_block_linearization_states,
    MarginalizationSummarySchur* summary) {
  // Validate the input. parameter_block_linearization_states will be validated
  // later.
  CHECK_NOTNULL(problem);
  for (double* parameter_block : parameter_blocks_to_marginalize) {
    CHECK(problem->HasParameterBlock(parameter_block))
        << "Parameter block to marginalize is not in the problem. Did you "
           "forget to add it?";
  }

  return internal::MarginalizeOutVariables(options,
                                           parameter_blocks_to_marginalize,
                                           problem,
                                           marginalization_prior_ids,
                                           parameter_block_linearization_states,
                                           summary);
}

bool MarginalizeOutVariablesQR(
    const MarginalizationOptionsQR& options,
    const std::set<double*>& parameter_blocks_to_marginalize,
    Problem* problem,
    std::vector<ResidualBlockId>* marginalization_prior_ids,
    const std::map<const double*, const double*>*
        parameter_block_linearization_states,
    MarginalizationSummaryQR* summary) {
  // Validate the input. parameter_block_linearization_states will be validated
  // later.
  CHECK_NOTNULL(problem);
  for (double* parameter_block : parameter_blocks_to_marginalize) {
    CHECK(problem->HasParameterBlock(parameter_block))
        << "Parameter block to marginalize is not in the problem. Did you "
           "forget to add it?";
  }

  return internal::MarginalizeOutVariables(options,
                                           parameter_blocks_to_marginalize,
                                           problem,
                                           marginalization_prior_ids,
                                           parameter_block_linearization_states,
                                           summary);
}

bool MarginalizeOutVariableAfterRebase(
    const MarginalizationOptionsSchur& marginalization_options,
    double* parameter_block_to_marginalize,
    const std::map<const double*, const double*>*
        parameter_block_linearization_states,
    const std::map<const double*, std::string>& parameter_block_names,
    const double** base_parameter_block,
    Problem* problem,
    std::vector<ResidualBlockId>* marginalization_prior_ids,
    std::set<const double*>* rebased_parameter_blocks,
    std::vector<std::unique_ptr<RebasedCostFunction>>* rebased_cost_functions) {
  if (!RebaseParameterBlockAndMarkovBlanket(parameter_block_names,
                                            parameter_block_to_marginalize,
                                            problem,
                                            base_parameter_block,
                                            rebased_parameter_blocks,
                                            rebased_cost_functions)) {
    return false;
  }

  return MarginalizeOutVariablesSchur(marginalization_options,
                                      {parameter_block_to_marginalize},
                                      problem,
                                      marginalization_prior_ids,
                                      parameter_block_linearization_states,
                                      /*summary=*/nullptr);
  return true;
}

}  // namespace ceres
