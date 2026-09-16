#include <gtest/gtest.h>

#include <gz/math/Helpers.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>

#include "dave_gz_model_plugins/SailJointTorqueDiagnostics.hh"

namespace ioes::sim
{
TEST(SailJointTorqueDiagnostics, ProjectsMomentAboutJointAxis)
{
  const gz::math::Pose3d jointPose(10, 20, 30, 0, 0, 0);

  const auto result = ComputeSailJointTorqueProjection(
    gz::math::Vector3d(10, 21, 30),
    gz::math::Vector3d(4, 0, 0),
    jointPose,
    gz::math::Vector3d(0, 0, 1));

  EXPECT_TRUE(result.valid);
  EXPECT_EQ(result.jointOriginWorld, gz::math::Vector3d(10, 20, 30));
  EXPECT_EQ(result.jointAxisWorld, gz::math::Vector3d(0, 0, 1));
  EXPECT_EQ(result.leverFromJointWorld, gz::math::Vector3d(0, 1, 0));
  EXPECT_EQ(result.momentAboutJointWorld, gz::math::Vector3d(0, 0, -4));
  EXPECT_DOUBLE_EQ(result.torqueAboutJointAxisNm, -4.0);
}

TEST(SailJointTorqueDiagnostics, UsesWorldRotatedJointAxis)
{
  const gz::math::Pose3d jointPose(0, 0, 0, 0, GZ_PI_2, 0);

  const auto result = ComputeSailJointTorqueProjection(
    gz::math::Vector3d(0, 1, 0),
    gz::math::Vector3d(0, 0, 3),
    jointPose,
    gz::math::Vector3d(0, 0, 1));

  ASSERT_TRUE(result.valid);
  EXPECT_NEAR(result.jointAxisWorld.X(), 1.0, 1e-12);
  EXPECT_NEAR(result.jointAxisWorld.Y(), 0.0, 1e-12);
  EXPECT_NEAR(result.jointAxisWorld.Z(), 0.0, 1e-12);
  EXPECT_NEAR(result.torqueAboutJointAxisNm, 3.0, 1e-12);
}

TEST(SailJointTorqueDiagnostics, RejectsZeroJointAxis)
{
  const auto result = ComputeSailJointTorqueProjection(
    gz::math::Vector3d(0, 1, 0),
    gz::math::Vector3d(1, 0, 0),
    gz::math::Pose3d::Zero,
    gz::math::Vector3d::Zero);

  EXPECT_FALSE(result.valid);
}
}  // namespace ioes::sim
