#include "dave_gz_model_plugins/LiftDragModel.hh"

#include <gz/common/Console.hh>
#include <gz/math/Helpers.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <sdf/Element.hh>

#include <chrono>
#include <memory>
#include <string>

namespace ioes::sim
{
namespace
{
template <typename T>
T GetParam(const std::shared_ptr<const sdf::Element> &_sdf,
           const std::string &_name,
           const T &_defaultValue)
{
  if (_sdf && _sdf->HasElement(_name))
    return _sdf->Get<T>(_name);
  return _defaultValue;
}
}

class SailLiftDragSystem:
  public gz::sim::System,
  public gz::sim::ISystemConfigure,
  public gz::sim::ISystemPreUpdate
{
public:
  void Configure(const gz::sim::Entity &_entity,
                 const std::shared_ptr<const sdf::Element> &_sdf,
                 gz::sim::EntityComponentManager &_ecm,
                 gz::sim::EventManager &/*_eventMgr*/) override
  {
    gzerr << "[SailLiftDragSystem] Configure called" << std::endl;

    this->model = gz::sim::Model(_entity);
    if (!this->model.Valid(_ecm))
    {
      gzerr << "[SailLiftDragSystem] Plugin must be attached to a model.\n";
      return;
    }

    this->linkName = GetParam<std::string>(_sdf, "link_name", "");
    if (this->linkName.empty())
      this->linkName = GetParam<std::string>(_sdf, "sail_link", "");

    gzerr << "[SailLiftDragSystem] sail_link = " << this->linkName << std::endl;

    if (this->linkName.empty())
    {
      gzerr << "[SailLiftDragSystem] Missing <link_name> or <sail_link>.\n";
      return;
    }

    const gz::sim::Entity linkEntity = this->model.LinkByName(_ecm, this->linkName);
    if (linkEntity == gz::sim::kNullEntity)
    {
      gzerr << "[SailLiftDragSystem] Link [" << this->linkName
            << "] not found. No sail forces will be generated.\n";
      return;
    }

    this->link = gz::sim::Link(linkEntity);
    this->link.EnableVelocityChecks(_ecm, true);

    LiftDragParams p;
    p.fluidDensity = GetParam<double>(_sdf, "fluid_density", 1.2);
    p.radialSymmetry = GetParam<bool>(_sdf, "radial_symmetry", true);
    p.forward = GetParam<gz::math::Vector3d>(_sdf, "forward", {1, 0, 0});
    p.upward = GetParam<gz::math::Vector3d>(_sdf, "upward", {0, 0, 1});
    p.cp = GetParam<gz::math::Vector3d>(_sdf, "cp", {0, 0, 0});
    p.area = GetParam<double>(_sdf, "area", 1.0);
    p.alpha0 = GetParam<double>(_sdf, "a0", 0.0);
    p.alphaStall = GetParam<double>(_sdf, "alpha_stall", GZ_PI / 4.0);
    p.cla = GetParam<double>(_sdf, "cla", 4.0 / GZ_PI);
    p.claStall = GetParam<double>(_sdf, "cla_stall", -4.0 / GZ_PI);
    p.cda = GetParam<double>(_sdf, "cda", 0.0);
    p.minSpeed = GetParam<double>(_sdf, "min_speed", 0.01);

    this->modelData.SetParams(p);
    this->windVelocityWorld =
      GetParam<gz::math::Vector3d>(_sdf, "wind_velocity", {0, 0, 0});
    this->debug = GetParam<bool>(_sdf, "debug", false);
    this->debugPeriod = GetParam<double>(_sdf, "debug_period", 1.0);

    gzmsg << "[SailLiftDragSystem] Loaded for link [" << this->linkName
          << "], wind_velocity=" << this->windVelocityWorld
          << ", area=" << p.area << ", rho=" << p.fluidDensity << "\n";
  }

  void PreUpdate(const gz::sim::UpdateInfo &_info,
                 gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused || this->link.Entity() == gz::sim::kNullEntity)
      return;

    const auto poseOpt = this->link.WorldPose(_ecm);
    const auto velOpt = this->link.WorldLinearVelocity(_ecm, this->modelData.Params().cp);
    if (!poseOpt || !velOpt)
      return;

    // Equivalent to asv_sim SailPlugin:
    // free stream = wind velocity at sail - sail CP velocity.
    const gz::math::Vector3d freeStream = this->windVelocityWorld - *velOpt;
    auto result = this->modelData.Compute(freeStream, *poseOpt);

    if (result.force.Length() <= 0.0)
      return;

    // Apply force at the center of pressure. The offset is in the link frame.
    this->link.AddWorldWrench(
      _ecm, result.force, gz::math::Vector3d::Zero, this->modelData.Params().cp);

    if (this->debug)
    {
      const double simSec = std::chrono::duration<double>(_info.simTime).count();
      if (simSec - this->lastDebugTime >= this->debugPeriod)
      {
        this->lastDebugTime = simSec;
        gzmsg << "[SailLiftDragSystem] link=" << this->linkName
              << " Vapp=" << freeStream
              << " speed=" << result.speed
              << " alpha_deg=" << (result.alpha * 180.0 / 3.14159265358979323846)
              << " cl=" << result.cl
              << " cd=" << result.cd
              << " lift=" << result.lift
              << " drag=" << result.drag
              << " force=" << result.force << "\n";
      }
    }
  }

private:
  gz::sim::Model model{gz::sim::kNullEntity};
  gz::sim::Link link{gz::sim::kNullEntity};
  std::string linkName;
  LiftDragModel modelData;
  gz::math::Vector3d windVelocityWorld{0, 0, 0};
  bool debug{false};
  double debugPeriod{1.0};
  double lastDebugTime{-1e9};
};
}  // namespace ioes::sim

GZ_ADD_PLUGIN(
  ioes::sim::SailLiftDragSystem,
  gz::sim::System,
  ioes::sim::SailLiftDragSystem::ISystemConfigure,
  ioes::sim::SailLiftDragSystem::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(
  ioes::sim::SailLiftDragSystem,
  "ioes::sim::SailLiftDragSystem")
