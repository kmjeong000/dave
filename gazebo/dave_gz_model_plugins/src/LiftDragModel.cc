#include "dave_gz_model_plugins/LiftDragModel.hh"

#include <algorithm>
#include <cmath>

namespace ioes::sim
{
namespace
{
gz::math::Vector3d SafeNormalize(const gz::math::Vector3d &_v)
{
  const double len = _v.Length();
  if (len <= 1e-12)
    return gz::math::Vector3d::Zero;
  return _v / len;
}

double Clamp(double _x, double _lo, double _hi)
{
  return std::max(_lo, std::min(_hi, _x));
}
}  // namespace

LiftDragModel::LiftDragModel(const LiftDragParams &_params)
{
  this->SetParams(_params);
}

void LiftDragModel::SetParams(const LiftDragParams &_params)
{
  this->params = _params;
  this->params.forward = SafeNormalize(this->params.forward);
  this->params.upward = SafeNormalize(this->params.upward);

  if (this->params.forward.Length() <= 1e-12)
    this->params.forward = gz::math::Vector3d(1, 0, 0);
  if (this->params.upward.Length() <= 1e-12)
    this->params.upward = gz::math::Vector3d(0, 0, 1);
}

const LiftDragParams &LiftDragModel::Params() const
{
  return this->params;
}

LiftDragResult LiftDragModel::Compute(
  const gz::math::Vector3d &_freeStreamVelocityWorld,
  const gz::math::Pose3d &_linkWorldPose) const
{
  LiftDragResult result;
  result.freeStreamVelocity = _freeStreamVelocityWorld;

  const double freeStreamSpeed = _freeStreamVelocityWorld.Length();
  if (freeStreamSpeed <= this->params.minSpeed)
    return result;

  const gz::math::Vector3d velUnit = _freeStreamVelocityWorld / freeStreamSpeed;

  const gz::math::Quaterniond rot = _linkWorldPose.Rot();
  const gz::math::Vector3d forwardWorld = SafeNormalize(rot.RotateVector(this->params.forward));

  gz::math::Vector3d upwardWorld;
  if (this->params.radialSymmetry)
  {
    // Same idea as asv_sim: for a symmetric foil/sail, construct a local
    // positive-lift reference direction from the incoming flow and chord.
    const gz::math::Vector3d tmp = forwardWorld.Cross(velUnit);
    upwardWorld = SafeNormalize(tmp.Cross(forwardWorld));
  }
  else
  {
    upwardWorld = SafeNormalize(rot.RotateVector(this->params.upward));
  }

  if (upwardWorld.Length() <= 1e-12)
    return result;

  const gz::math::Vector3d spanWorld = SafeNormalize(forwardWorld.Cross(upwardWorld));
  if (spanWorld.Length() <= 1e-12)
    return result;

  // Project the free-stream velocity into the lift-drag plane.
  // This removes the component parallel to the span direction.
  const gz::math::Vector3d velLD =
    _freeStreamVelocityWorld - _freeStreamVelocityWorld.Dot(spanWorld) * spanWorld;

  const double speedLD = velLD.Length();
  if (speedLD <= this->params.minSpeed)
    return result;

  const gz::math::Vector3d dragUnit = velLD / speedLD;
  const gz::math::Vector3d liftUnit = SafeNormalize(dragUnit.Cross(spanWorld));

  const double cosAlpha = Clamp(-forwardWorld.Dot(dragUnit), -1.0, 1.0);
  const double alpha = std::acos(cosAlpha);

  const double q = 0.5 * this->params.fluidDensity * speedLD * speedLD;
  const double cl = this->LiftCoefficient(alpha);
  const double cd = this->DragCoefficient(alpha);

  result.alpha = alpha;
  result.speed = speedLD;
  result.cl = cl;
  result.cd = cd;
  result.lift = cl * q * this->params.area * liftUnit;
  result.drag = cd * q * this->params.area * dragUnit;
  result.force = result.lift + result.drag;

  return result;
}

// Lift is piecewise-linear and symmetric about alpha = pi / 2, matching
// the asv_sim LiftDragModel behavior.
double LiftDragModel::LiftCoefficient(double _alpha) const
{
  const double alpha0 = this->params.alpha0;
  const double cla = this->params.cla;
  const double alphaStall = this->params.alphaStall;
  const double claStall = this->params.claStall;

  const auto f1 = [=](double _x) { return cla * (_x - alpha0); };
  const auto f2 = [=](double _x)
  {
    if (_x < alphaStall)
      return f1(_x);
    return claStall * (_x - alphaStall) + f1(alphaStall);
  };

  if (_alpha < GZ_PI / 2.0)
    return f2(_alpha);
  return f2(GZ_PI - _alpha);
}

// Drag is piecewise-linear and symmetric about alpha = pi / 2, matching
// the asv_sim LiftDragModel behavior.
double LiftDragModel::DragCoefficient(double _alpha) const
{
  const double cda = this->params.cda;
  const auto f1 = [=](double _x) { return cda * _x; };

  if (_alpha < GZ_PI / 2.0)
    return f1(_alpha);
  return f1(GZ_PI - _alpha);
}
}  // namespace ioes::sim
