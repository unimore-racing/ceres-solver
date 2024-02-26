// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2023 Google Inc. All rights reserved.
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
// Author: sameeragarwal@google.com (Sameer Agarwal)
//
// This example fits the curve f(x;m,c) = e^(m * x + c) to data, minimizing the
// sum squared loss.

#include <ceres/problem.h>
#include <ceres/sized_cost_function.h>
#include <ceres/types.h>
#include <sys/types.h>

#include <array>
#include <chrono>
#include <memory_resource>
#include <vector>

#include "ceres/ceres.h"
#include "glog/logging.h"


// clang-format on

class ExponentialResidual : public ceres::SizedCostFunction<1, 1, 1> {
 public:
  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const {
    return true;
  }

  static inline std::array<uint8_t, 1000000 * 128> buffer{};
  static inline std::pmr::monotonic_buffer_resource buffer_res{buffer.data(),
                                                               buffer.size()};
  static inline std::pmr::unsynchronized_pool_resource allocator{&buffer_res};
  static void* operator new(size_t size) { return allocator.allocate(size); }

  static void operator delete(void* ptr, size_t size) {
    return allocator.deallocate(ptr, size);
  }
};

typedef ExponentialResidual Res;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  const double initial_m = 0.0;
  const double initial_c = 0.0;
  double m = initial_m;
  double c = initial_c;

  ceres::Problem::Options opts;
  opts.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  opts.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  // opts.enable_fast_removal = false;
  opts.disable_all_safety_checks = true;
  ceres::Problem problem(opts);
  std::vector<ceres::ResidualBlockId> residuals;
  residuals.reserve(100000);

  problem.AddParameterBlock(&m, 1);
  problem.AddParameterBlock(&c, 1);

  std::chrono::steady_clock::time_point begin, end;
  begin = std::chrono::steady_clock::now();
  for (int i = 0; i < 50000; ++i) {
    auto res = new Res();
    auto id = problem.AddResidualBlock(res, nullptr, &m, &c);
    residuals.push_back(id);
  }
  end = std::chrono::steady_clock::now();
  std::cout << "Add = "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     begin)
                   .count()
            << "[us]" << std::endl;

  begin = std::chrono::steady_clock::now();
  for (int i = 0; i < 50000; ++i) {
    problem.RemoveResidualBlock(residuals.back());
    residuals.pop_back();
  }
  end = std::chrono::steady_clock::now();
  std::cout << "Remove = "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     begin)
                   .count()
            << "[us]" << std::endl;

  begin = std::chrono::steady_clock::now();
  for (int i = 0; i < 50000; ++i) {
    auto res = new Res();
    auto id = problem.AddResidualBlock(res, nullptr, &m, &c);
    residuals.push_back(id);
  }
  end = std::chrono::steady_clock::now();
  std::cout << "Add = "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     begin)
                   .count()
            << "[us]" << std::endl;

  ceres::Solver::Options options;
  options.max_num_iterations = 25;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  // ceres::Solver::Summary summary;
  // ceres::Solve(options, &problem, &summary);
  // std::cout << summary.BriefReport() << "\n";
  // std::cout << "Initial m: " << initial_m << " c: " << initial_c << "\n";
  // std::cout << "Final   m: " << m << " c: " << c << "\n";
  return 0;
}
