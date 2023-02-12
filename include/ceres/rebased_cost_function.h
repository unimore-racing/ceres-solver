#ifndef CERES_INTERNAL_REBASED_COST_FUNCTION_H_
#define CERES_INTERNAL_REBASED_COST_FUNCTION_H_

#include <map>
#include <optional>
#include <vector>

#include "ceres/cost_function.h"
#include "ceres/internal/eigen.h"
#include "ceres/marginalizable_manifold.h"

namespace ceres {

class RebasedCostFunction : public CostFunction {
 public:
  // Base block is not one of the blocks in the original cost function.
  // It should be the last in the array of parameter block pointers passed to
  // evaluate
  RebasedCostFunction(
      const CostFunction& original_cost_function,
      const std::map<const double*, std::string>& parameter_block_names,
      const std::vector<bool>& parameter_block_is_rebased,
      const LieGroup& manifold)
      : original_cost_function_had_base_parameter_block_(false),
        parameter_block_names_(parameter_block_names),
        parameter_block_is_rebased_(parameter_block_is_rebased),
        original_cost_function_(original_cost_function),
        manifold_(manifold) {
    CHECK(original_cost_function_.parameter_block_sizes().size() + 1 ==
          parameter_block_is_rebased.size());
    CHECK(parameter_block_is_rebased.back() == false);
    CHECK(!parameter_block_is_rebased.empty());

    CHECK(!parameter_block_is_rebased.back()) << "Expected the last parameter block to be the base";
    size_t num_rebased = 0;
    for (size_t i = 0 ; i < parameter_block_is_rebased.size() - 1; ++i)
    {
      if (parameter_block_is_rebased[i])
      {
        ++num_rebased;
      }
    }
    CHECK(num_rebased > 0) << "Expected at least one rebased parameter block!";

    // Set num residuals
    set_num_residuals(original_cost_function_.num_residuals());

    // Set sizes
    for (const int32_t size : original_cost_function_.parameter_block_sizes()) {
      mutable_parameter_block_sizes()->push_back(size);
    }
    mutable_parameter_block_sizes()->push_back(manifold.AmbientSize());
    idx_base_ = parameter_block_sizes().size() - 1;
  }

  // The base block was in the original cost function at index idx_base, so
  // we have the same number of parameter blocks in the rebased cost function.
  RebasedCostFunction(
      const CostFunction& original_cost_function,
      const std::map<const double*, std::string>& parameter_block_names,
      const std::vector<bool>& parameter_block_is_rebased,
      const LieGroup& manifold,
      int32_t idx_base)
      : original_cost_function_had_base_parameter_block_(true),
        parameter_block_names_(parameter_block_names),
        parameter_block_is_rebased_(parameter_block_is_rebased),
        original_cost_function_(original_cost_function),
        manifold_(manifold),
        idx_base_(idx_base) {
    // Set num residuals
    set_num_residuals(original_cost_function.num_residuals());

    CHECK(!parameter_block_is_rebased[idx_base]) << "parameter_block_is_rebased[idx_base] should be false";
    size_t num_rebased = 0;
    for (int i = 0 ; i < parameter_block_is_rebased.size(); ++i)
    {
      if (idx_base != i && parameter_block_is_rebased[i])
      {
        ++num_rebased;
      }
    }
    CHECK(num_rebased > 0);

    // Set sizes
    for (const int32_t size : original_cost_function.parameter_block_sizes()) {
      mutable_parameter_block_sizes()->push_back(size);
    }
    CHECK(idx_base_ < parameter_block_sizes().size());
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const override {
    const int ambient_size = manifold_.AmbientSize();
    const int num_parameter_blocks = parameter_block_sizes().size();
    const int num_parameter_blocks_original =
        original_cost_function_had_base_parameter_block_
            ? num_parameter_blocks
            : num_parameter_blocks - 1;

    for (int i = 0 ; i < num_parameter_blocks ; ++i)
    {
      for (int j = i+1 ; j < num_parameter_blocks ; ++j)
      {
        CHECK(parameters[i] != parameters[j]) << "Parameter blocks cannot alias";
      }
    }

    const double* const parameter_block_base = parameters[idx_base_];

    std::vector<Vector> world_frame_parameter_blocks;
    std::vector<const double*> world_frame_parameter_blocks_ptrs_for_orig_cost;

    // The world-frame parameter block is computed as
    // (x in world) = (base) * (x in local)

    for (int32_t i = 0; i < num_parameter_blocks; ++i) {

      if (parameter_block_is_rebased_.at(i))
      {
        // This parameter block has been rebased. Re-represent it in the
        // original world.
        Vector& wf_pb = world_frame_parameter_blocks.emplace_back(ambient_size);
        world_frame_parameter_blocks_ptrs_for_orig_cost.push_back(wf_pb.data());
        manifold_.Compose(parameter_block_base, parameters[i], wf_pb.data());
      }
      else
      {
        if (!original_cost_function_had_base_parameter_block_ && i == idx_base_)
        {
          // This was appended to the original cost function, so we do not need to use it in evaluating the original cost function.
          CHECK(idx_base_ == num_parameter_blocks - 1);
        }
        else
        {
          world_frame_parameter_blocks_ptrs_for_orig_cost.push_back(parameters[i]);
        }
      }
    }

    for (const double* pb_ptr : world_frame_parameter_blocks_ptrs_for_orig_cost)
    {
      CHECK(pb_ptr) << "Nullptr in world_frame_parameter_blocks_ptrs_for_orig_cost!";
    }

    if (jacobians == nullptr) {
      // Evaluate the original cost function
      original_cost_function_.Evaluate(
          world_frame_parameter_blocks_ptrs_for_orig_cost.data(), residuals, nullptr);
      return true;
    }

    // Jacobians
    std::vector<Matrix> original_jacobians_scratch;
    std::vector<double*> orig_cost_func_jacobians_ptrs(
        num_parameter_blocks_original);
    // Prepare the Jacobian pointers for the original cost function
    for (int32_t i = 0; i < num_parameter_blocks_original; ++i) {
      if (!parameter_block_is_rebased_.at(i) ||
          (original_cost_function_had_base_parameter_block_ &&
           i == idx_base_)) {
        // The original Jacobian is the desired jacobian for this block.
        orig_cost_func_jacobians_ptrs.at(i) = jacobians[i];
      } else {
        // Use scratch to store the Jacobian for the original cost function.
        Matrix& wf_J = original_jacobians_scratch.emplace_back(num_residuals(),
                                                               ambient_size);
        orig_cost_func_jacobians_ptrs.at(i) = wf_J.data();
      }
    }

    // Evaluate using the original cost function and Jacobians.
    original_cost_function_.Evaluate(world_frame_parameter_blocks_ptrs_for_orig_cost.data(),
                                     residuals,
                                     orig_cost_func_jacobians_ptrs.data());


    // Chain the original Jacobian with Plus and Minus Jacobians.
    MatrixRef jacobian_wrt_base(
        jacobians[idx_base_], num_residuals(), ambient_size);
    if (!original_cost_function_had_base_parameter_block_) {
      jacobian_wrt_base.setZero();
    }

    for (int32_t block_idx = 0; block_idx < num_parameter_blocks_original;
         ++block_idx) {
      if (!parameter_block_is_rebased_.at(block_idx)) {
        // We already have the Jacobian from the previous call to Evaluate.
        continue;
      }

      // d/dx r(base * x) = Jr * d/dx(base * x)
      ConstMatrixRef original_jacobian(orig_cost_func_jacobians_ptrs.at(block_idx),
                                       num_residuals(),
                                       ambient_size);

      Matrix compose_jacobian_wrt_pb(ambient_size, ambient_size);
      Matrix compose_jacobian_wrt_base(ambient_size, ambient_size);
      manifold_.ComposeJacobian(parameters[idx_base_],
                                parameters[block_idx],
                                compose_jacobian_wrt_base.data(),
                                compose_jacobian_wrt_pb.data());
      jacobian_wrt_base += original_jacobian * compose_jacobian_wrt_base;

      MatrixRef jacobian_wrt_pb(jacobians[block_idx], num_residuals(), ambient_size);
      jacobian_wrt_pb = original_jacobian * compose_jacobian_wrt_pb;
    }

    return true;
  }

 private:
  const bool original_cost_function_had_base_parameter_block_;
  const std::map<const double*, std::string> parameter_block_names_;
  const std::vector<bool> parameter_block_is_rebased_;
  const CostFunction& original_cost_function_;
  const LieGroup& manifold_;
  int idx_base_;
};

}  // namespace ceres

#endif  // CERES_INTERNAL_REBASED_COST_FUNCTION_H_
