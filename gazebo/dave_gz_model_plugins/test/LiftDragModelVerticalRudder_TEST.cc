#include "dave_gz_model_plugins/LiftDragModel.hh"

#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include <gz/math/Pose3.hh>
#include <gz/math/Quaternion.hh>
#include <gz/math/Vector3.hh>

namespace
{
constexpr double kPi = 3.14159265358979323846;

double Radians(double _degrees)
{
  return _degrees * kPi / 180.0;
}

double Degrees(double _radians)
{
  return _radians * 180.0 / kPi;
}

ioes::sim::LiftDragParams RudderParams(
  bool _radialSymmetry,
  double _lateralReferenceSign = 1.0)
{
  ioes::sim::LiftDragParams params;
  params.fluidDensity = 1000.0;
  params.radialSymmetry = _radialSymmetry;
  params.forward.Set(0.0, 1.0, 0.0);
  params.upward.Set(_lateralReferenceSign, 0.0, 0.0);
  params.area = 0.30;
  params.alpha0 = 0.0;
  params.cla = 0.8;
  params.alphaStall = 0.3;
  params.claStall = -0.3;
  params.cda = 0.05;
  params.minSpeed = 0.01;
  return params;
}

void PrintResult(
  const char *_label,
  const ioes::sim::LiftDragResult &_result)
{
  std::cout << std::fixed << std::setprecision(6)
            << _label
            << ": alpha_deg=" << Degrees(_result.alpha)
            << ", force_n=(" << _result.force.X()
            << ", " << _result.force.Y()
            << ", " << _result.force.Z() << ")\n";
}
}  // namespace

TEST(LiftDragModelVerticalRudder,
     ProductionFixedPlaneProducesLateralForceAndPositiveYawMoment)
{
  constexpr double waterSpeedMps = 2.0;
  const double rudderYawRad = Radians(5.0);
  const double hullPitchRad = Radians(3.5);

  // The sailboat model points along +Y, so physical pitch is a rotation about
  // the model's lateral +X axis.  The relative flow remains longitudinal and
  // horizontal in world coordinates.
  const gz::math::Quaterniond baseRotation(hullPitchRad, 0.0, 0.0);
  const gz::math::Quaterniond rudderRelativeRotation(
    0.0, 0.0, rudderYawRad);
  const gz::math::Pose3d rudderPose(
    gz::math::Vector3d::Zero,
    baseRotation * rudderRelativeRotation);
  const gz::math::Vector3d relativeWaterFlowWorld(
    0.0, -waterSpeedMps, 0.0);

  const ioes::sim::LiftDragModel radialComparisonModel(
    RudderParams(true));
  const auto radialComparison = radialComparisonModel.Compute(
    relativeWaterFlowWorld, rudderPose);

  // Production vertical-rudder convention: fixed horizontal lift plane,
  // +Y chord direction, and +X positive-lift reference.
  const ioes::sim::LiftDragModel productionModel(
    RudderParams(false, 1.0));
  const auto production = productionModel.Compute(
    relativeWaterFlowWorld, rudderPose);

  PrintResult("radial_symmetry_comparison", radialComparison);
  PrintResult("production_fixed_lateral_+X", production);

  // With radial symmetry the model constructs its lift plane from the full
  // 3-D flow vector.  Both rudder yaw and hull pitch therefore contribute to
  // alpha, and the pitch contribution produces a substantial vertical force.
  const double expectedRadialAlpha = std::acos(
    std::cos(hullPitchRad) * std::cos(rudderYawRad));
  EXPECT_NEAR(radialComparison.alpha, expectedRadialAlpha, 1e-12);
  EXPECT_GT(
    std::abs(radialComparison.force.Z()),
    std::abs(radialComparison.force.Y()));
  EXPECT_GT(
    std::abs(radialComparison.force.Z()),
    0.4 * radialComparison.force.Length());

  // A fixed vertical-rudder lift plane removes the spanwise pitch component
  // before calculating alpha.  Its alpha is set primarily by the 5-degree
  // yaw, and either lateral reference sign keeps the force overwhelmingly in
  // the horizontal lateral direction.  The sign selects the positive-lift
  // direction without changing that plane.
  EXPECT_NEAR(production.alpha, rudderYawRad, 1e-12);
  EXPECT_GT(radialComparison.alpha, production.alpha);
  EXPECT_GT(production.force.X(), 0.0);
  EXPECT_GT(
    production.force.X(),
    10.0 * std::abs(production.force.Y()));
  EXPECT_LT(
    std::abs(production.force.Z()),
    0.02 * std::abs(production.force.X()));
  EXPECT_GT(
    std::abs(radialComparison.force.Z()),
    10.0 * std::abs(production.force.Z()));

  // The production CP is aft of the base origin.  Project r x F onto the
  // hull's actual world yaw axis, matching FoilLiftDragSystem diagnostics.
  const gz::math::Vector3d rearCpLocal(0.0, -1.48, -0.08);
  const gz::math::Vector3d rearCpWorld =
    rudderPose.Rot().RotateVector(rearCpLocal);
  const gz::math::Vector3d momentAboutBaseWorld =
    rearCpWorld.Cross(production.force);
  const gz::math::Vector3d yawAxisWorld =
    baseRotation.RotateVector(gz::math::Vector3d(0.0, 0.0, 1.0));
  const double yawMomentNm = momentAboutBaseWorld.Dot(yawAxisWorld);
  std::cout << "production_rear_cp_yaw_moment_nm="
            << yawMomentNm << "\n";
  EXPECT_GT(yawMomentNm, 0.0);
}
