#include "dave_gz_model_plugins/VirtualSheetActuator.hh"

#include <gtest/gtest.h>

#include <cmath>

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kMaximumAllowance = kPi / 4.0;
constexpr double kLinearStiffness = 300.0;
constexpr double kQuadraticStiffness = 460.0;
constexpr double kDamping = 44.0;
constexpr double kMaximumTorque = 260.0;
constexpr double kContainmentLimit = 15.0 * kPi / 180.0;
constexpr double kTakeUpRate = 30.0 * kPi / 180.0;

ioes::sim::VirtualSheetActuatorResult Compute(
    const double _position,
    const double _velocity,
    const double _allowance)
{
  return ioes::sim::ComputeVirtualSheetActuator(
    _position, _velocity, _allowance, kMaximumAllowance,
    kLinearStiffness, kQuadraticStiffness, kDamping, kMaximumTorque);
}
}  // namespace

TEST(VirtualSheetAllowanceState, ReleasesSheetOutImmediately)
{
  const auto result = ioes::sim::UpdateVirtualSheetAllowance(
    20.0 * kPi / 180.0, 40.0 * kPi / 180.0, 0.0,
    kMaximumAllowance, kContainmentLimit, kTakeUpRate, 0.1);

  EXPECT_NEAR(result.effectiveAllowanceRad, 40.0 * kPi / 180.0, 1e-12);
  EXPECT_DOUBLE_EQ(result.lagRad, 0.0);
  EXPECT_TRUE(result.sheetOutRelease);
  EXPECT_FALSE(result.tightening);
}

TEST(VirtualSheetAllowanceState, TightensAtFiniteNominalRate)
{
  const auto result = ioes::sim::UpdateVirtualSheetAllowance(
    45.0 * kPi / 180.0, 15.0 * kPi / 180.0, 0.0,
    kMaximumAllowance, kContainmentLimit, kTakeUpRate, 0.1);

  EXPECT_NEAR(result.effectiveAllowanceRad, 42.0 * kPi / 180.0, 1e-12);
  EXPECT_NEAR(result.lagRad, 27.0 * kPi / 180.0, 1e-12);
  EXPECT_TRUE(result.tightening);
  EXPECT_FALSE(result.containmentViolation);
}

TEST(VirtualSheetAllowanceState, DoesNotTightenPastBoomSafetyFloor)
{
  const auto result = ioes::sim::UpdateVirtualSheetAllowance(
    40.0 * kPi / 180.0, 10.0 * kPi / 180.0,
    55.0 * kPi / 180.0, kMaximumAllowance,
    kContainmentLimit, kTakeUpRate, 0.1);

  EXPECT_NEAR(result.safeFloorRad, 40.0 * kPi / 180.0, 1e-12);
  EXPECT_NEAR(result.effectiveAllowanceRad, 40.0 * kPi / 180.0, 1e-12);
  EXPECT_FALSE(result.containmentViolation);
}

TEST(VirtualSheetAllowanceState, DoesNotFollowOutwardBoomToHideViolation)
{
  const auto result = ioes::sim::UpdateVirtualSheetAllowance(
    40.0 * kPi / 180.0, 10.0 * kPi / 180.0,
    56.0 * kPi / 180.0, kMaximumAllowance,
    kContainmentLimit, kTakeUpRate, 0.1);

  EXPECT_NEAR(result.safeFloorRad, 41.0 * kPi / 180.0, 1e-12);
  EXPECT_NEAR(result.effectiveAllowanceRad, 40.0 * kPi / 180.0, 1e-12);
  EXPECT_TRUE(result.containmentViolation);
}

TEST(VirtualSheetAllowanceState, ConvergesToFixedReachableRawCommand)
{
  double effective = 45.0 * kPi / 180.0;
  const double raw = 30.0 * kPi / 180.0;
  for (int i = 0; i < 5; ++i)
  {
    effective = ioes::sim::UpdateVirtualSheetAllowance(
      effective, raw, 20.0 * kPi / 180.0, kMaximumAllowance,
      kContainmentLimit, kTakeUpRate, 0.1).effectiveAllowanceRad;
  }

  EXPECT_NEAR(effective, raw, 1e-12);
}

TEST(VirtualSheetActuator, IsExactlySlackInsideCommandedAllowance)
{
  for (const double allowance : {0.0, 0.2, kMaximumAllowance})
  {
    for (const double side : {-1.0, 1.0})
    {
      const auto inside = Compute(side * 0.5 * allowance, side * 2.0, allowance);
      const auto boundary = Compute(side * allowance, side * 2.0, allowance);
      EXPECT_FALSE(inside.active);
      EXPECT_FALSE(boundary.active);
      EXPECT_DOUBLE_EQ(inside.torqueNm, 0.0);
      EXPECT_DOUBLE_EQ(boundary.torqueNm, 0.0);
      EXPECT_DOUBLE_EQ(boundary.penetrationRad, 0.0);
    }
  }
}

TEST(VirtualSheetActuator, ClampsAllowanceToUnsignedOperationalRange)
{
  EXPECT_DOUBLE_EQ(Compute(0.0, 0.0, -0.2).allowanceRad, 0.0);
  EXPECT_DOUBLE_EQ(
    Compute(0.0, 0.0, 2.0 * kMaximumAllowance).allowanceRad,
    kMaximumAllowance);
}

TEST(VirtualSheetActuator, RestoresBothSidesAndDampsOnlyOutwardMotion)
{
  constexpr double allowance = 0.3;
  constexpr double penetration = 0.1;
  const double spring =
    kLinearStiffness * penetration +
    kQuadraticStiffness * penetration * penetration;
  const double damping = kDamping * 0.2;

  const auto positiveOutward = Compute(allowance + penetration, 0.2, allowance);
  const auto negativeOutward = Compute(-allowance - penetration, -0.2, allowance);
  const auto positiveInward = Compute(allowance + penetration, -0.2, allowance);

  EXPECT_NEAR(positiveOutward.torqueNm, -(spring + damping), 1e-12);
  EXPECT_NEAR(negativeOutward.torqueNm, spring + damping, 1e-12);
  EXPECT_NEAR(positiveInward.torqueNm, -spring, 1e-12);
  EXPECT_NEAR(positiveOutward.torqueNm, -negativeOutward.torqueNm, 1e-12);
  EXPECT_DOUBLE_EQ(positiveInward.dampingTorqueNm, 0.0);
}

TEST(VirtualSheetActuator, AppliesProgressiveQuadraticSpring)
{
  constexpr double allowance = kMaximumAllowance;
  for (const auto &[boomAngleDeg, expectedTorqueNm] : {
         std::pair{50.0, 29.68},
         std::pair{55.0, 66.37},
         std::pair{60.0, 110.07}})
  {
    const auto result = Compute(boomAngleDeg * kPi / 180.0, 0.0, allowance);
    EXPECT_NEAR(result.springTorqueNm, expectedTorqueNm, 0.01);
    EXPECT_NEAR(result.torqueNm, -expectedTorqueNm, 0.01);
  }
}

TEST(VirtualSheetActuator, CapsFiniteRestoringTorque)
{
  const auto positive = Compute(kPi / 2.0, 10.0, 0.0);
  const auto negative = Compute(-kPi / 2.0, -10.0, 0.0);

  EXPECT_TRUE(positive.capped);
  EXPECT_TRUE(negative.capped);
  EXPECT_DOUBLE_EQ(positive.torqueNm, -kMaximumTorque);
  EXPECT_DOUBLE_EQ(negative.torqueNm, kMaximumTorque);
}

TEST(VirtualSheetActuator, SpringIsContinuousAtAllowanceBoundaryAtRest)
{
  constexpr double allowance = 0.4;
  constexpr double epsilon = 1e-9;
  for (const double side : {-1.0, 1.0})
  {
    const auto atBoundary = Compute(side * allowance, 0.0, allowance);
    const auto justOutside = Compute(
      side * (allowance + epsilon), 0.0, allowance);
    EXPECT_DOUBLE_EQ(atBoundary.torqueNm, 0.0);
    EXPECT_LT(std::abs(justOutside.torqueNm), 1e-6);
  }
}
