#pragma once

#include <gz/math/Helpers.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>

namespace ioes::sim
{
struct LiftDragParams
{
  double fluidDensity{1.2};
  bool radialSymmetry{true};

  // Body/link-frame directions.
  // forward: foil chord direction.
  // upward: positive-lift reference direction when radialSymmetry is false.
  gz::math::Vector3d forward{1, 0, 0};
  gz::math::Vector3d upward{0, 0, 1};

  // Center of pressure in the link frame.
  gz::math::Vector3d cp{0, 0, 0};

  double area{1.0};
  double alpha0{0.0};
  double cla{4.0 / GZ_PI};
  double alphaStall{GZ_PI / 4.0};
  double claStall{-4.0 / GZ_PI};
  double cda{0.0};

  double minSpeed{0.01};
};

struct LiftDragResult
{
  gz::math::Vector3d lift{0, 0, 0};
  gz::math::Vector3d drag{0, 0, 0};
  gz::math::Vector3d force{0, 0, 0};
  gz::math::Vector3d torque{0, 0, 0};

  gz::math::Vector3d freeStreamVelocity{0, 0, 0};
  double alpha{0.0};
  double speed{0.0};
  double cl{0.0};
  double cd{0.0};
};

class LiftDragModel
{
public:
  explicit LiftDragModel(const LiftDragParams &_params = LiftDragParams());

  void SetParams(const LiftDragParams &_params);
  const LiftDragParams &Params() const;

  LiftDragResult Compute(
    const gz::math::Vector3d &_freeStreamVelocityWorld,
    const gz::math::Pose3d &_linkWorldPose) const;

  double LiftCoefficient(double _alpha) const;
  double DragCoefficient(double _alpha) const;

private:
  LiftDragParams params;
};
}  // namespace ioes::sim
