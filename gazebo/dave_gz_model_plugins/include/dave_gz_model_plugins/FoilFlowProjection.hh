#pragma once

#include <gz/math/Vector3.hh>

namespace ioes::sim
{
inline gz::math::Vector3d ProjectVectorOntoPlane(
  const gz::math::Vector3d &_vector,
  const gz::math::Vector3d &_planeNormal)
{
  const double normalLength = _planeNormal.Length();
  if (normalLength <= 1e-12)
    return _vector;

  const gz::math::Vector3d normal = _planeNormal / normalLength;
  return _vector - _vector.Dot(normal) * normal;
}
}  // namespace ioes::sim
