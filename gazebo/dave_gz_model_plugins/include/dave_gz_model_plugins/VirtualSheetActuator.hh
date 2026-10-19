#pragma once

#include <algorithm>
#include <cmath>

namespace ioes::sim
{
struct VirtualSheetActuatorResult
{
  double allowanceRad{0.0};
  double penetrationRad{0.0};
  double outwardVelocityRadS{0.0};
  double springTorqueNm{0.0};
  double dampingTorqueNm{0.0};
  double requestedTorqueNm{0.0};
  double torqueNm{0.0};
  bool active{false};
  bool capped{false};
};

struct VirtualSheetAllowanceStateResult
{
  double rawAllowanceRad{0.0};
  double effectiveAllowanceRad{0.0};
  double lagRad{0.0};
  double safeFloorRad{0.0};
  bool sheetOutRelease{false};
  bool tightening{false};
  bool containmentViolation{false};
};

/// Advance the persistent reduced-order sheet-length state.
///
/// A larger allowance pays sheet out and is released immediately. A smaller
/// allowance is taken up at a finite rate, but never tightened past the
/// current boom-position safety floor. If outward boom motion has already
/// crossed that floor, the boundary is held rather than moved outward to hide
/// the physical containment violation.
inline VirtualSheetAllowanceStateResult UpdateVirtualSheetAllowance(
    const double _effectiveAllowanceRad,
    const double _rawAllowanceRad,
    const double _boomPositionRad,
    const double _maximumAllowanceRad,
    const double _containmentLimitRad,
    const double _takeUpRateRadS,
    const double _dtS)
{
  VirtualSheetAllowanceStateResult result;
  const double maximumAllowanceRad = std::max(_maximumAllowanceRad, 0.0);
  const double containmentLimitRad = std::max(_containmentLimitRad, 0.0);
  const double takeUpRateRadS = std::max(_takeUpRateRadS, 0.0);
  const double dtS = std::max(_dtS, 0.0);

  result.rawAllowanceRad = std::clamp(
    _rawAllowanceRad, 0.0, maximumAllowanceRad);
  const double currentEffectiveAllowanceRad = std::clamp(
    _effectiveAllowanceRad, 0.0, maximumAllowanceRad);
  result.safeFloorRad = std::max(
    std::abs(_boomPositionRad) - containmentLimitRad, 0.0);

  if (result.rawAllowanceRad >= currentEffectiveAllowanceRad)
  {
    result.effectiveAllowanceRad = result.rawAllowanceRad;
    result.sheetOutRelease =
      result.rawAllowanceRad > currentEffectiveAllowanceRad;
  }
  else
  {
    result.tightening = true;
    const double desiredAllowanceRad = std::max(
      result.rawAllowanceRad,
      currentEffectiveAllowanceRad - takeUpRateRadS * dtS);

    if (result.safeFloorRad <= currentEffectiveAllowanceRad)
    {
      result.effectiveAllowanceRad = std::max(
        desiredAllowanceRad, result.safeFloorRad);
    }
    else
    {
      result.effectiveAllowanceRad = currentEffectiveAllowanceRad;
    }
  }

  result.effectiveAllowanceRad = std::clamp(
    result.effectiveAllowanceRad,
    result.rawAllowanceRad,
    maximumAllowanceRad);
  result.lagRad =
    result.effectiveAllowanceRad - result.rawAllowanceRad;
  result.containmentViolation =
    std::max(
      std::abs(_boomPositionRad) - result.effectiveAllowanceRad, 0.0) >
    containmentLimitRad + 1e-12;
  return result;
}

/// Compute the torque from a reduced-order, unilateral virtual sheet.
///
/// The unsigned allowance is the maximum sheet-out angle. The sheet is slack
/// inside that envelope and can only pull the boom inward outside it. Positive
/// torque increases the signed sail-joint angle.
inline VirtualSheetActuatorResult ComputeVirtualSheetActuator(
    const double _positionRad,
    const double _velocityRadS,
    const double _commandedAllowanceRad,
    const double _maximumAllowanceRad,
    const double _linearStiffnessNmPerRad,
    const double _quadraticStiffnessNmPerRad2,
    const double _dampingNmPerRadS,
    const double _maximumTorqueNm)
{
  VirtualSheetActuatorResult result;
  result.allowanceRad = std::clamp(
    _commandedAllowanceRad, 0.0, _maximumAllowanceRad);

  const double side = _positionRad < 0.0 ? -1.0 : 1.0;
  result.penetrationRad =
    std::max(std::abs(_positionRad) - result.allowanceRad, 0.0);
  result.outwardVelocityRadS = side * _velocityRadS;

  if (result.penetrationRad <= 0.0)
    return result;

  result.springTorqueNm =
    _linearStiffnessNmPerRad * result.penetrationRad +
    _quadraticStiffnessNmPerRad2 *
      result.penetrationRad * result.penetrationRad;
  result.dampingTorqueNm =
    _dampingNmPerRadS * std::max(result.outwardVelocityRadS, 0.0);
  result.requestedTorqueNm =
    result.springTorqueNm + result.dampingTorqueNm;

  const double magnitude = std::clamp(
    result.requestedTorqueNm, 0.0, _maximumTorqueNm);
  result.torqueNm = -side * magnitude;
  result.active = magnitude > 0.0;
  result.capped = result.requestedTorqueNm > _maximumTorqueNm;
  return result;
}
}  // namespace ioes::sim
