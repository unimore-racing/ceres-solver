// Author: evanlevine138e@gmail.com (Evan Levine)

#ifndef CERES_PUBLIC_REBASE_H_
#define CERES_PUBLIC_REBASE_H_

#include <memory>
#include <vector>

#include "ceres/internal/eigen.h"
#include "ceres/problem.h"
#include "ceres/rebased_cost_function.h"

namespace ceres {

// Changes the parameterization of parameter_block to be relative to
// some arbitrarily chosen parameter block in Markov blanket, and changes the
// other variables in its Markov blanket to to be relative to this parameter
// block as well. If successful, base_parameter_block points to the base
// parameter block.
bool RebaseParameterBlockAndMarkovBlanket(
    const std::map<const double*, std::string>& parameter_block_names,
    const double* parameter_block,
    Problem* problem,
    const double** base_pb_output,
    std::set<const double*>* rebased_parameter_blocks,
    std::vector<std::unique_ptr<RebasedCostFunction>>* rebased_cost_functions);

}  // namespace ceres

#endif  // CERES_PUBLIC_REBASE_H_
