#include "dave_gz_model_plugins/FoilFlowProjection.hh"
#include "dave_gz_model_plugins/LiftDragModel.hh"

#include <gtest/gtest.h>

#include <cmath>

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

ioes::sim::LiftDragParams Stage1RudderParams()
{
  ioes::sim::LiftDragParams params;
  params.fluidDensity = 1000.0;
  params.radialSymmetry = true;
  params.forward.Set(0.0, 1.0, 0.0);
  params.upward.Set(0.0, 0.0, 1.0);
  params.cp.Set(0.0, -1.48, -0.08);
  params.area = 0.30;
  params.alpha0 = 0.0;
  params.cla = 0.8;
  params.alphaStall = 0.3;
  params.claStall = -0.3;
  params.cda = 0.05;
  params.minSpeed = 0.01;
  return params;
}

double YawMoment(
  const gz::math::Vector3d &_applicationPointWorld,
  const gz::math::Vector3d &_forceWorld,
  const gz::math::Vector3d &_yawAxisWorld)
{
  return _applicationPointWorld.Cross(_forceWorld).Dot(_yawAxisWorld);
}
}  // namespace

TEST(FoilFlowProjection, RemovesSpanFlowAndPreservesStage1YawSign)
{
  const double hullPitchRad = Radians(3.5);
  const gz::math::Quaterniond hullRotation(hullPitchRad, 0.0, 0.0);
  const gz::math::Vector3d yawAxisWorld =
    hullRotation.RotateVector(gz::math::Vector3d(0.0, 0.0, 1.0));
  const gz::math::Vector3d freeStreamWorld(0.0, -2.0, 0.0);
  const gz::math::Vector3d projectedFreeStream =
    ioes::sim::ProjectVectorOntoPlane(freeStreamWorld, yawAxisWorld);

  EXPECT_GT(std::abs(freeStreamWorld.Dot(yawAxisWorld)), 0.1);
  EXPECT_NEAR(projectedFreeStream.Dot(yawAxisWorld), 0.0, 1e-12);

  const ioes::sim::LiftDragParams params = Stage1RudderParams();
  const ioes::sim::LiftDragModel model(params);
  for (const double rudderDegrees : {5.0, -5.0})
  {
    const gz::math::Quaterniond rudderRotation =
      hullRotation *
      gz::math::Quaterniond(0.0, 0.0, Radians(rudderDegrees));
    const gz::math::Pose3d rudderPose(
      gz::math::Vector3d::Zero, rudderRotation);
    const auto result = model.Compute(projectedFreeStream, rudderPose);

    EXPECT_NEAR(result.force.Dot(yawAxisWorld), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(result.alpha), Radians(5.0), 1e-12);
    EXPECT_NEAR(result.force.Dot(result.freeStreamVelocity),
                result.drag.Dot(result.freeStreamVelocity), 1e-12);

    const gz::math::Vector3d applicationPointWorld =
      rudderPose.Rot().RotateVector(params.cp);
    const double yawMoment =
      YawMoment(applicationPointWorld, result.force, yawAxisWorld);
    if (rudderDegrees > 0.0)
      EXPECT_LT(yawMoment, 0.0);
    else
      EXPECT_GT(yawMoment, 0.0);
  }
}

TEST(FoilFlowProjection, ZeroNormalLeavesFlowUnchanged)
{
  const gz::math::Vector3d input(1.0, 2.0, 3.0);
  EXPECT_EQ(
    ioes::sim::ProjectVectorOntoPlane(input, gz::math::Vector3d::Zero),
    input);
}
