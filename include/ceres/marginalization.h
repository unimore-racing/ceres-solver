// Author: evanlevine138e@gmail.com (Evan Levine)

#ifndef CERES_PUBLIC_MARGINALIZATION_H_
#define CERES_PUBLIC_MARGINALIZATION_H_

#include <map>
#include <set>
#include <vector>

#include "ceres/problem.h"
#include "ceres/rebased_cost_function.h"

namespace ceres {

// Background
// ==========
//
// Marginalization enables solving a problem for a subset of variables of
// interest at lower computational cost compared to solving the original
// problem. It requires making a linear approximation of the residuals with
// respect to the parameters to be marginalized out and the parameters that
// separate these variables from the rest of the graph, called the
// Markov blanket. The variables to marginalize out are replaced by an error
// term involving the variables in the Markov blanket. See [1], [2].
//
// The parameterization for the variables in the Markov blanket should be chosen
// judiciously. For example, accuracy may degrade if the second derivative of
// the residuals with respect to the tangent-space increment is large for the
// Markov blanket variables or for the variables to marginalize out.
//
// Consider a robustified non-linear least squares problem
//
// min_x 0.5 \sum_{i} \rho_i(\|f_i(x_i_1, ..., x_i_k)\|^2)
// s.t. l_j ≤ x_j ≤ u_j
//
// We can partition the variables into the variables to marginalize out,
// denoted m, the variables related to them by error terms (their Markov
// blanket), denoted b, and the remaining variables r. Suppose that the
// bounds l_j and u_j are trivial, -infinity and +infinity respectively, x_j in
// m.
//
// min_x 0.5 \sum_{i in dM} \rho_i(\|f_i(b, m)\|^2) +
//       0.5 \sum_{i not in dM} \rho_i(\|f_i(b, r)\|^2),
//
// where dM is the index set of all error terms involving m. Let b^0 and
// m^0 be linearization points for b and m to be respectively and [+] be
// the boxed plus operator. We can then make the following linear approximation
// for the first term.
//
// c(b, \delta m) = 0.5\sum_{i in dM} \rho_i(\|f_i(b, m^0 [+] \delta m)\|^2)
//                  ~ 0.5\sum_{i in dM} \rho_i(\|f_i(b^0, m^0) +
//                                            J_i [\delta b ; \delta m]\|^2),
// where J_i = [ df_i/db db/d\delta b,  df_i/dm dm/d\delta m], ";"
// denotes vertical concatenation, and \delta m is the error state for m =
// m^0 [+] \delta m.
//
// c(b,\delta m) = (g^T + [\delta b; \delta m]^T H) [\delta b; \delta m],
// where g = \sum_i \rho' J_i^T f_i(b^0, m^0),
//  H = \sum_i \rho' J_i^T J_i.
//
// Partition H into the block matrix
//  H = [ H_{mm}  H_{bm}^T ]
//      [ H_{bm}  H_{bb}   ].
// and g into the block vector g = [g_{mm}; g_{mb}].
//
// Minimize c(\delta b, \delta m) with respect to \delta m:
//
// argmin_{\delta m} c(\delta b, \delta m) =
//    H_{mm}^-1 (g_{mm} +  H_{mb}(\delta b))
//
// Substitution into c yields
//
// g_t^T(\delta b) + 0.5(\delta b) H_t(\delta b) + |f|^2,
//
// where  H_t = H_{bb} -  H_{bm} H_{mm}^{-1} H_{bm}^T
//        g_t = g_{mb} -  H_{bm} H_{mm}^{-1} g_{mm}.
//
// We can write this as
//
// \|D^(1/2) U^T \delta b + D^(-1/2) U^T g_t\|^2,
//
// where H_t = U * D * U^T is the eigen-decomposition of H_t with D
// containing only the nonzero eigenvalues on the diagonal. This is the cost
// function for the "marginalization prior" to be added to the graph with the
// marginalized parameter blocks removed. Alternatively, a Cholesky
// factorization can be used if it is known that H_t is full-rank.
//
// In this implementation, H_{mm} is assumed to be dense.
//
// [1] Carlevaris-Bianco, Nicholas, Michael Kaess, and Ryan M. Eustice. "Generic
// node removal for factor-graph SLAM." IEEE Transactions on Robotics 30.6
// (2014): 1371-1385.
//
// [2] Eckenhoff, Kevin, Liam Paull, and Guoquan Huang.
// "Decoupled, consistent node removal and edge sparsification for graph-based
// SLAM." 2016 IEEE/RSJ International Conference on Intelligent Robots and
// Systems (IROS). IEEE, 2016.

struct MarginalizationOptionsQR {
  // For "QR," this is the rank threshold in the QR factorization.
  double rank_threshold = 1e-10;

  bool compute_jacobian_condition_number = false;
};

struct MarginalizationOptionsSchur {
  enum struct Method {
    Eigendecomposition = 0,
    LDLT = 1,
    ModifiedLDLT = 2,
  };

  Method method = Method::Eigendecomposition;

  // For LDLT and Eigendecomposition, indicates whether to assume
  // that the block of the information matrix corresponding to marginalized
  // variables is full rank ( H_{mm}).
  bool assume_marginalized_block_is_full_rank = false;

  // Compute the condition number of the jacobian [Jb, Jm] for the problem containing the blocks to marginalize and their blanket.
  bool compute_jacobian_condition_number = false;

  // For "Eigendecomposition," this is the eigenvalue threshold for
  // determining the rank of the Jacobian for the marginalization prior.
  double rank_threshold = 1e-10;

  // For "LDLT," control the number of attempts made in computing the
  // pseudoinverse of the information for the marginal. This should be
  // used with the understanding that it adds a measurement that injects
  // spurious information. The number of attempts is limited to at most
  // max_ldlt_failure_fix_attempts.
  size_t max_ldlt_failure_fix_attempts = 0;
};

struct MarginalizationSummarySchur
{
  std::optional<double> condition_number_jacobian;

  // Valid if Method::Eigendecomposition is used.
  std::optional<double> condition_number_marginal_information;

  // Valid if Method::ModifiedLDLT or Method::LDLT is used.
  // This is the Frobenius norm of the perturbation of \Lambda_t used in the factorization.
  double ldlt_perturbation_norm;
};

struct MarginalizationSummaryQR
{
  std::optional<double> condition_number_jacobian;
};

// Marginalize out a set of variables. If the computation fails, returns false
// and does not modify the problem. If the computations succeeds, removes the
// variables to marginalize, adds a linear cost function for the marginalization
// prior and returns true. If marginalization_prior_id is not null, the residual
// block for the marginalization prior is returned in it. Optionally,
// linearization points used for Jacobians can be provided in
// parameter_block_linearization_states, a mapping from user pointers to the
// parameter blocks to pointers to values to be used for linearization.
[[nodiscard]] bool MarginalizeOutVariablesQR(
    const MarginalizationOptionsQR& options,
    const std::set<double*>& parameter_blocks_to_marginalize,
    Problem* problem,
    std::vector<ResidualBlockId>* marginalization_prior_ids,
    const std::map<const double*, const double*>*
        parameter_block_linearization_states = nullptr,
    MarginalizationSummaryQR* summary = nullptr);

[[nodiscard]] bool MarginalizeOutVariablesSchur(
    const MarginalizationOptionsSchur& options,
    const std::set<double*>& parameter_blocks_to_marginalize,
    Problem* problem,
    std::vector<ResidualBlockId>* marginalization_prior_ids,
    const std::map<const double*, const double*>*
        parameter_block_linearization_states = nullptr,
    MarginalizationSummarySchur* summary = nullptr);

[[nodiscard]] bool MarginalizeOutVariableAfterRebase(
    const MarginalizationOptionsSchur& options,
    double* parameter_block_to_marginalize,
    const std::map<const double*, const double*>*
        parameter_block_linearization_states,
    const std::map<const double*, std::string>& parameter_block_names,
    const double** base_parameter_block,
    Problem* problem,
    std::vector<ResidualBlockId>* marginalization_prior_ids,
    std::set<const double*>* rebased_parameter_blocks,
    std::vector<std::unique_ptr<RebasedCostFunction>>* rebased_cost_functions);

}  // namespace ceres

#endif  // CERES_PUBLIC_MARGINALIZATION_H_
