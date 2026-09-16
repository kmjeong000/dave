#pragma once

#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>

namespace ioes::sim
{
struct SailJointTorqueProjection
{
  gz::math::Vector3d jointOriginWorld{0, 0, 0};
  gz::math::Vector3d jointAxisWorld{0, 0, 0};
  gz::math::Vector3d leverFromJointWorld{0, 0, 0};
  gz::math::Vector3d momentAboutJointWorld{0, 0, 0};
  double torqueAboutJointAxisNm{0.0};
  bool valid{false};
};

inline SailJointTorqueProjection ComputeSailJointTorqueProjection(
    const gz::math::Vector3d &_cpWorld,
    const gz::math::Vector3d &_appliedForceWorld,
    const gz::math::Pose3d &_jointWorldPose,
    const gz::math::Vector3d &_jointAxisLocal)
{
  SailJointTorqueProjection result;
  result.jointOriginWorld = _jointWorldPose.Pos();

  const double axisLength = _jointAxisLocal.Length();
  if (axisLength <= 1e-12)
    return result;

  result.jointAxisWorld =
    _jointWorldPose.Rot().RotateVector(_jointAxisLocal / axisLength);
  result.leverFromJointWorld = _cpWorld - result.jointOriginWorld;
  result.momentAboutJointWorld =
    result.leverFromJointWorld.Cross(_appliedForceWorld);
  result.torqueAboutJointAxisNm =
    result.momentAboutJointWorld.Dot(result.jointAxisWorld);
  result.valid = true;
  return result;
}
}  // namespace ioes::sim
