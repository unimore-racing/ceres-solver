// Author: evanlevine138e@gmail.com (Evan Levine)

#include <bitset>
#include <memory>
#include <numeric>

#include "ceres/manifold_test_utils.h"
#include "ceres/marginalizable_manifold.h"
#include "ceres/pose2manifold.h"
#include "ceres/rotation.h"
#include "ceres/so2manifold.h"

namespace ceres {
namespace internal {

static Vector MakeRandomUnitVector(int d) {
  Vector w = Vector::Random(d);
  w = w / w.norm();
  return w;
}

static Vector MakeRandomSO2Storage() { return MakeRandomUnitVector(2); }

static Vector MakeRandomQuaternionStorage() { return MakeRandomUnitVector(4); }

static Vector MakeRandomSO3Storage() {
  Vector mat(9);
  const Vector w = Vector::Random(3);
  AngleAxisToRotationMatrix(w.data(), mat.data());
  return mat;
}

static Vector MakeRandomPose2Storage() {
  const Vector xR = MakeRandomUnitVector(2);
  Vector x = Vector::Random(4);
  x.head(2) = xR;
  return x;
}

static Vector MakeRandomSE2Storage() {
  const Vector rot = MakeRandomUnitVector(2);
  const Vector pos = Vector::Random(2);
  Vector x(4);
  x.head(2) = rot;
  x.tail(2) = pos;
  return x;
}

template <typename ManifoldType>
void TestComposeBetweenRoundTrip(const ManifoldType& manifold,
                                 const Vector& x,
                                 const Vector& base) {
  Vector x_rebased(manifold.AmbientSize());
  manifold.Between(base.data(), x.data(), x_rebased.data());
  Vector y(manifold.AmbientSize());
  manifold.Compose(base.data(), x_rebased.data(), y.data());
  // y = x_rebased + base
  // x_rebased = x - base
  // y = x
  const double errorNorm = (x - y).norm();
  ASSERT_LT(errorNorm, 1e-14);
}

template <typename ManifoldType>
void TestComposeJacobian(const ManifoldType& manifold,
                         const Vector& w,
                         const Vector& x) {
  static constexpr double kStepSize = 1e-8;
  static constexpr double kInvStepSize = 1.0 / kStepSize;

  const int ambient_size = manifold.AmbientSize();
  Vector wx(ambient_size);
  manifold.Compose(w.data(), x.data(), wx.data());

  Matrix dwx_dw(ambient_size, ambient_size);
  Matrix dwx_dx(ambient_size, ambient_size);
  manifold.ComposeJacobian(w.data(), x.data(), dwx_dw.data(), dwx_dx.data());

  Matrix dwx_dx_numerical(ambient_size, ambient_size);
  for (int i = 0; i < ambient_size; i++) {
    Vector x_plus_step = x;
    x_plus_step[i] += kStepSize;
    Vector prod_step(ambient_size);
    manifold.Compose(w.data(), x_plus_step.data(), prod_step.data());
    const Vector dwx_dxi = kInvStepSize * (prod_step - wx);
    dwx_dx_numerical.col(i) = dwx_dxi;
  }

  Matrix dwx_dw_numerical(ambient_size, ambient_size);
  for (int i = 0; i < ambient_size; i++) {
    Vector w_plus_step = w;
    w_plus_step[i] += kStepSize;
    Vector prod_step(ambient_size);
    manifold.Compose(w_plus_step.data(), x.data(), prod_step.data());
    const Vector dwx_dwi = kInvStepSize * (prod_step - wx);
    dwx_dw_numerical.col(i) = dwx_dwi;
  }

  /*
  std::cout << "dwx_dx" << std::endl;
  std::cout << dwx_dx << std::endl;
  std::cout << "dwx_dx_numerical" << std::endl;
  std::cout << dwx_dx_numerical << std::endl;
  std::cout << "dwx_dw" << std::endl;
  std::cout << dwx_dw << std::endl;
  std::cout << "dwx_dw_numerical" << std::endl;
  std::cout << dwx_dw_numerical << std::endl;
  */

  const Matrix err_w = dwx_dw - dwx_dw_numerical;
  const Matrix err_x = dwx_dx - dwx_dx_numerical;
  ASSERT_LT(err_w.norm(), 1e-5);
  ASSERT_LT(err_x.norm(), 1e-5);
}

template <typename ManifoldType>
void CheckMarginalizableManifold(const ManifoldType& manifold,
                                 const Vector& x,
                                 const Vector& y) {
  const int ambient_size = manifold.AmbientSize();
  const int tan_size = manifold.TangentSize();

  Matrix JMinus(tan_size, ambient_size);
  manifold.MinusJacobian2(x.data(), y.data(), JMinus.data());
  Matrix JPlus(ambient_size, tan_size);
  manifold.PlusJacobian(x.data(), JPlus.data());

  static constexpr double kStep = 1e-7;

  Matrix lhs_numerical(tan_size, tan_size);
  for (int i = 0; i < tan_size; ++i) {
    auto compute_lhs_numerical = [&](double stepSize) -> Vector {
      // Step is ei * stepSize
      Vector delta(tan_size);
      delta.setZero();
      delta[i] = stepSize;

      Vector x_plus_delta(ambient_size);
      manifold.Plus(x.data(), delta.data(), x_plus_delta.data());

      Vector out(tan_size);
      manifold.Minus(x_plus_delta.data(), y.data(), out.data());
      return out;
    };

    lhs_numerical.col(i) =
        (compute_lhs_numerical(kStep) - compute_lhs_numerical(-kStep)) /
        (2.0 * kStep);
  }

  const Matrix rhs = JMinus * JPlus;
  const Matrix error = lhs_numerical - rhs;

  /*
  std::cout << "J+\n";
  std::cout << JPlus << std::endl;
  std::cout << "J-\n";
  std::cout << JMinus << std::endl;
  std::cout << "error\n";
  std::cout << error << std::endl;
  std::cout << "lhs_numerical\n";
  std::cout << lhs_numerical << std::endl;
  std::cout << "rhs\n";
  std::cout << rhs << std::endl;
  */

  ASSERT_LT(error.norm(), 1e-8);
}

TEST(MarginalizableManifold, TestMarginalizableSO3Manifold) {
  static constexpr int kAmbientSize = 9;
  static constexpr int kTangentSize = 3;
  MarginalizableSO3Manifold manifold;
  EXPECT_EQ(manifold.AmbientSize(), kAmbientSize);
  EXPECT_EQ(manifold.TangentSize(), kTangentSize);
  static constexpr int kNumTrials = 10;
  static constexpr double kTolerance = 1e-9;

  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = MakeRandomSO3Storage();
    const Vector y = MakeRandomSO3Storage();
    const Vector delta = Vector::Random(kTangentSize);

    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y, kTolerance);

    ASSERT_NO_FATAL_FAILURE(CheckMarginalizableManifold(manifold, y, y));
    ASSERT_NO_FATAL_FAILURE(CheckMarginalizableManifold(manifold, x, y));
  }
}

// TODO evlevine clean up these tests

#if 0
TEST(MarginalizableManifold, TestMarginalizableQuaternionManifold) {
  static constexpr int kAmbientSize = 4;
  static constexpr int kTangentSize = 3;
  MarginalizableQuaternionManifold manifold;
  EXPECT_EQ(manifold.AmbientSize(), kAmbientSize);
  EXPECT_EQ(manifold.TangentSize(), kTangentSize);
  static constexpr int kNumTrials = 10;
  static constexpr double kTolerance = 1e-9;

  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = MakeRandomQuaternionStorage();
    const Vector y = MakeRandomQuaternionStorage();
    const Vector delta = Vector::Random(kTangentSize);

    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y, kTolerance);

    ASSERT_NO_FATAL_FAILURE(CheckMarginalizableManifold(manifold, y, y));
    ASSERT_NO_FATAL_FAILURE(CheckMarginalizableManifold(manifold, x, y));
  }
}
#endif

TEST(MarginalizableManifold, EuclideanManifold) {
  static constexpr int kAmbientSize = 3;
  static constexpr int kTangentSize = 3;
  MarginalizableEuclideanManifold<kTangentSize> manifold;
  EXPECT_EQ(manifold.AmbientSize(), kAmbientSize);
  EXPECT_EQ(manifold.TangentSize(), kTangentSize);
  static constexpr int kNumTrials = 10;
  static constexpr double kTolerance = 1e-9;

  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = Vector::Random(kAmbientSize);
    const Vector y = Vector::Random(kAmbientSize);
    const Vector delta = Vector::Random(kTangentSize);

    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y, kTolerance);

    ASSERT_NO_FATAL_FAILURE(CheckMarginalizableManifold(manifold, y, y));
    ASSERT_NO_FATAL_FAILURE(CheckMarginalizableManifold(manifold, x, y));
  }
}

template <typename ManifoldType, int kTangentSize, int kAmbientSize>
static void TestLieGroupManifold(std::function<Vector()> makeRandomInstance) {
  ManifoldType manifold;
  EXPECT_EQ(manifold.AmbientSize(), kAmbientSize);
  EXPECT_EQ(manifold.TangentSize(), kTangentSize);
  static constexpr int kNumTrials = 10;
  static constexpr double kTolerance = 1e-9;

  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = makeRandomInstance();
    const Vector y = makeRandomInstance();
    const Vector delta = Vector::Random(kTangentSize);

    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y, kTolerance);

    ASSERT_NO_FATAL_FAILURE(CheckMarginalizableManifold(manifold, y, y));
    ASSERT_NO_FATAL_FAILURE(CheckMarginalizableManifold(manifold, x, y));
  }

  {
    const Vector base = makeRandomInstance();
    const Vector x = makeRandomInstance();
    TestComposeBetweenRoundTrip(manifold, x, base);
  }

  {
    const Vector w = makeRandomInstance();
    const Vector x = makeRandomInstance();
    TestComposeJacobian(manifold, w, x);
  }
}

TEST(MarginalizableManifold, SO2Manifold) {
  TestLieGroupManifold<SO2Manifold, 1, 2>(MakeRandomSO2Storage);
}

TEST(MarginalizableManifold, Pose2Manifold) {
  TestLieGroupManifold<Pose2Manifold, 3, 4>(MakeRandomPose2Storage);
}

TEST(MarginalizableManifold, EuclideanComposeBetweenRoundTrip) {
  static constexpr size_t kAmbientSize = 4;
  const EuclideanLieGroup<kAmbientSize> manifold;
  const Vector base = Vector::Random(kAmbientSize);
  const Vector x = Vector::Random(kAmbientSize);
  TestComposeBetweenRoundTrip(manifold, x, base);
}

TEST(MarginalizableManifold, EuclideanComposeJacobian) {
  static constexpr size_t kAmbientSize = 4;
  const EuclideanLieGroup<kAmbientSize> manifold;
  const Vector w = Vector::Random(kAmbientSize);
  const Vector x = Vector::Random(kAmbientSize);
  TestComposeJacobian(manifold, w, x);
}

TEST(MarginalizableManifold, Pose2ComposeJacobian) {
  Pose2Manifold manifold;
  const Vector w = MakeRandomPose2Storage();
  const Vector x = MakeRandomPose2Storage();
  TestComposeJacobian(manifold, w, x);
}

TEST(MarginalizableManifold, QuaternionComposeJacobian) {
  QuaternionLieGroup manifold;
  const Vector w = MakeRandomQuaternionStorage();
  const Vector x = MakeRandomQuaternionStorage();
  TestComposeJacobian(manifold, w, x);
}

TEST(MarginalizableManifold, ToRotationMatrix) {
  using namespace liegroups;
  const Vector r = MakeRandomSO2Storage();
  const double cr = r[0];
  const double sr = r[1];
  SO2<double> R_so2(r.data());
  Matrix R(2, 2);
  for (int i = 0; i < 2; ++i) {
    Vector ek(2);
    ek.setZero();
    ek[i] = 1.0;
    Vector transformed(2);
    transform_point(transformed.data(), R_so2, ek.data());
    R.col(i) = transformed;
  }

  const Matrix R2 = to_rotation_matrix(R_so2);
  ASSERT_NEAR(R(0, 0), R2(0, 0), 1e-7);
  ASSERT_NEAR(R(0, 1), R2(0, 1), 1e-7);
  ASSERT_NEAR(R(1, 0), R2(1, 0), 1e-7);
  ASSERT_NEAR(R(1, 1), R2(1, 1), 1e-7);
}

}  // namespace internal
}  // namespace ceres
