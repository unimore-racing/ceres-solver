
// Author: evanlevine138e@gmail.com (Evan Levine)

#include "ceres/rebase.h"

#include <set>
#include <vector>

#include "ceres/rebased_cost_function.h"

namespace ceres {

namespace {
double* FindBaseParameterBlockNear(const double* parameter_block,
                                   Problem* problem) {
  std::vector<ResidualBlockId> residual_blocks;
  problem->GetResidualBlocksForParameterBlock(parameter_block,
                                              &residual_blocks);

  using Pair_t = std::pair<ResidualBlockId, double>;
  std::vector<Pair_t> residual_blocks_cost;
  for (const ResidualBlockId& residual_block_id : residual_blocks) {
    std::vector<double*> parameter_blocks_for_res;
    problem->GetParameterBlocksForResidualBlock(residual_block_id,
                                                &parameter_blocks_for_res);

    double residual_block_cost;
    if (!problem->EvaluateResidualBlock(residual_block_id, true, &residual_block_cost, nullptr, nullptr))
    {
      residual_block_cost = std::numeric_limits<double>::max();
    }
    residual_blocks_cost.push_back({residual_block_id, residual_block_cost});
  }

  std::sort(
      residual_blocks_cost.begin(),
      residual_blocks_cost.end(),
      [](const Pair_t& a, const Pair_t& b) { return a.second < b.second; });

  for (const auto& [residual_block_id, _] : residual_blocks_cost)
  {
    std::vector<double*> parameter_blocks_for_res;
    problem->GetParameterBlocksForResidualBlock(residual_block_id,
                                                &parameter_blocks_for_res);

    // Add any new parameter blocks for this residual block.
    std::vector<bool> parameter_block_is_rebased(
        parameter_blocks_for_res.size());
    bool residual_block_already_has_base_pb = false;
    int32_t residual_pb_idx_for_base = 0;
    for (size_t pb_for_res_idx = 0;
         pb_for_res_idx < parameter_blocks_for_res.size();
         ++pb_for_res_idx) {
      double* pb_for_res_curr = parameter_blocks_for_res.at(pb_for_res_idx);
      CHECK(!problem->IsParameterBlockConstant(pb_for_res_curr))
          << "Constant parameter blocks are not currently supported yet";

      if (parameter_block != pb_for_res_curr) {
        return pb_for_res_curr;
      }
    }
  }

  // Could not find one.
  return nullptr;
}
}  // namespace

// Changes the parameterization of parameter_block to be relative to
// some parameter block in Markov blanket, and change the other variables in its
// Markov blanket to to be relative to this parameter block as well. If
// successful, pb_base points to the base parameter block.
// Must use DO_NOT_TAKE_OWNERSHIP
bool RebaseParameterBlockAndMarkovBlanket(
    const std::map<const double*, std::string>& parameter_block_names,
    const double* parameter_block,
    Problem* problem,
    const double** pb_base_output,
    std::set<const double*>* rebased_parameter_blocks,
    std::vector<std::unique_ptr<RebasedCostFunction>>* rebased_cost_functions) {
  CHECK_NOTNULL(parameter_block);
  CHECK_NOTNULL(problem);
  CHECK_NOTNULL(rebased_parameter_blocks);
  CHECK_NOTNULL(rebased_cost_functions);

  bool pb_base_is_valid = false;
  std::vector<ResidualBlockId> blanket_residual_blocks;
  std::set<ResidualBlockId> residuals_visited;
  const auto* manifold =
      dynamic_cast<const LieGroup*>(problem->GetManifold(parameter_block));
  CHECK_NOTNULL(manifold);
  double* pb_base = FindBaseParameterBlockNear(parameter_block, problem);
  if (pb_base_output)
  {
    *pb_base_output = pb_base;
  }
  if (pb_base == nullptr) {
    return false;
  }
  problem->GetResidualBlocksForParameterBlock(parameter_block,
                                              &blanket_residual_blocks);

  // Get parameter blocks to rebase.
  std::set<double*> parameter_blocks_to_rebase;
  for (const ResidualBlockId& residual_block_id : blanket_residual_blocks) {
    std::vector<double*> parameter_blocks_for_res;
    problem->GetParameterBlocksForResidualBlock(residual_block_id,
                                                &parameter_blocks_for_res);
    for (double* pb : parameter_blocks_for_res) {
      if (pb_base != pb) {
        parameter_blocks_to_rebase.insert(pb);
      }
    }
  }

  // Get residual blocks to rebase.
  std::set<ResidualBlockId> residual_blocks_to_rebase;
  for (const double* pb : parameter_blocks_to_rebase) {
    std::vector<ResidualBlockId> pb_blanket_residual_blocks;
    problem->GetResidualBlocksForParameterBlock(pb,
                                                &pb_blanket_residual_blocks);

    for (const ResidualBlockId id : pb_blanket_residual_blocks) {
      residual_blocks_to_rebase.insert(id);
    }
  }

  // Rebase all required residual blocks.
  for (const ResidualBlockId& residual_block_id : residual_blocks_to_rebase) {
    std::vector<double*> parameter_blocks_for_res;
    problem->GetParameterBlocksForResidualBlock(residual_block_id,
                                                &parameter_blocks_for_res);

    // Add any new parameter blocks for this residual block.
    std::vector<bool> parameter_block_is_rebased;
    bool residual_block_already_has_base_pb = false;
    int32_t residual_pb_idx_for_base = 0;
    for (size_t pb_for_res_idx = 0;
         pb_for_res_idx < parameter_blocks_for_res.size();
         ++pb_for_res_idx) {
      double* pb_for_res_curr = parameter_blocks_for_res[pb_for_res_idx];
      CHECK(!problem->IsParameterBlockConstant(pb_for_res_curr))
          << "Constant parameter blocks are not currently supported yet";

      // problem->GetResidualBlocksForParameterBlock(pb_for_res_curr,
      // &bb_residual_blocks);

      // Rebased parameter blocks holds blocks that WILL be rebased below.
      parameter_block_is_rebased.push_back(parameter_blocks_to_rebase.find(pb_for_res_curr) != parameter_blocks_to_rebase.end());
      if (pb_base == pb_for_res_curr) {
        residual_block_already_has_base_pb = true;
        residual_pb_idx_for_base = pb_for_res_idx;
      }
    }

    const CostFunction* cost_function =
        problem->GetCostFunctionForResidualBlock(residual_block_id);
    LossFunction* loss_function = const_cast<LossFunction*>(
        problem->GetLossFunctionForResidualBlock(residual_block_id));

    // TODO ownership of the cost function and loss function?
    std::vector<double*> parameter_blocks_in_new_residual_block =
        parameter_blocks_for_res;
    if (residual_block_already_has_base_pb) {
      rebased_cost_functions->push_back(
          std::make_unique<RebasedCostFunction>(*cost_function,
                                                parameter_block_names,
                                                parameter_block_is_rebased,
                                                *manifold,
                                                residual_pb_idx_for_base));
    } else {

      // The last parameter block will be the base. Add it.
      parameter_blocks_in_new_residual_block.push_back(pb_base);
      parameter_block_is_rebased.push_back(false);
      
      // Create the cost function
      rebased_cost_functions->push_back(
          std::make_unique<RebasedCostFunction>(*cost_function,
                                                parameter_block_names,
                                                parameter_block_is_rebased,
                                                *manifold));
    }

    // The new problem is for evaluation. The manifold, loss function, and
    // cost function metadata will not be modified.
    CostFunction* cost_function_base = rebased_cost_functions->back().get();

    problem->AddResidualBlock(cost_function_base,
                              loss_function,
                              parameter_blocks_in_new_residual_block);
  }

  // Remove the original residual blocks. TODO evlevine: do not delete!!!!
  for (const ResidualBlockId& id : residual_blocks_to_rebase)
  {
    problem->RemoveResidualBlock(id);
  }

  // Update the states for the parameter blocks that we are rebasing.
  rebased_parameter_blocks->clear();
  for (double* pb : parameter_blocks_to_rebase) {
    const Vector pb_orig = ConstVectorRef(pb, manifold->AmbientSize());
    manifold->Between(pb_base, pb_orig.data(), pb);
    rebased_parameter_blocks->insert(pb);
  }

  return true;
}

}  // namespace ceres