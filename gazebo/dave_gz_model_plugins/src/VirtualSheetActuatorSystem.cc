#include "dave_gz_model_plugins/VirtualSheetActuator.hh"

#include <gz/common/Console.hh>
#include <gz/msgs/double.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Joint.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/components/JointForceCmd.hh>
#include <gz/transport/Node.hh>
#include <sdf/Element.hh>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <sstream>
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
}  // namespace

/// Reduced-order mainsheet actuator.
///
/// The command is an unsigned maximum sheet-out angle, not a signed position
/// target. This system is the sole writer of sail_joint JointForceCmd: zero
/// force while slack, and inward-only spring-damper force while taut.
class VirtualSheetActuatorSystem:
  public gz::sim::System,
  public gz::sim::ISystemConfigure,
  public gz::sim::ISystemPreUpdate
{
public:
  void Configure(
      const gz::sim::Entity &_entity,
      const std::shared_ptr<const sdf::Element> &_sdf,
      gz::sim::EntityComponentManager &_ecm,
      gz::sim::EventManager & /*_eventManager*/) override
  {
    this->model = gz::sim::Model(_entity);
    if (!this->model.Valid(_ecm))
    {
      gzerr << "[VirtualSheetActuator] Plugin must be attached to a model.\n";
      return;
    }

    this->jointName = GetParam<std::string>(_sdf, "joint_name", "sail_joint");
    this->baseLinkName = GetParam<std::string>(_sdf, "base_link", "base_link");
    this->topic = GetParam<std::string>(
      _sdf, "topic", "/model/sailboat/joint/sail_joint/cmd_pos");
    this->maximumAllowanceRad = GetParam<double>(
      _sdf, "maximum_allowance_rad", 0.7854);
    const double initialAllowanceRad = GetParam<double>(
      _sdf, "initial_allowance_rad", this->maximumAllowanceRad);
    this->linearStiffnessNmPerRad = GetParam<double>(
      _sdf, "linear_stiffness_nm_per_rad", 300.0);
    this->quadraticStiffnessNmPerRad2 = GetParam<double>(
      _sdf, "quadratic_stiffness_nm_per_rad2", 460.0);
    this->dampingNmPerRadS = GetParam<double>(
      _sdf, "damping_nm_per_rad_s", 44.0);
    this->maximumTorqueNm = GetParam<double>(
      _sdf, "maximum_torque_nm", 260.0);
    this->containmentLimitRad = GetParam<double>(
      _sdf, "containment_limit_rad", 0.2617993877991494);
    this->takeUpRateRadS = GetParam<double>(
      _sdf, "take_up_rate_rad_s", 0.5235987755982988);
    this->physicsRateTelemetry = GetParam<bool>(
      _sdf, "physics_rate_telemetry", true);

    if (this->topic.empty() || this->maximumAllowanceRad <= 0.0 ||
        this->linearStiffnessNmPerRad < 0.0 ||
        this->quadraticStiffnessNmPerRad2 < 0.0 ||
        this->dampingNmPerRadS < 0.0 || this->maximumTorqueNm <= 0.0 ||
        this->containmentLimitRad <= 0.0 || this->takeUpRateRadS <= 0.0)
    {
      gzerr << "[VirtualSheetActuator] Invalid configuration.\n";
      return;
    }

    this->commandedAllowanceRad.store(std::clamp(
      initialAllowanceRad, 0.0, this->maximumAllowanceRad));
    this->effectiveAllowanceRad = this->commandedAllowanceRad.load();

    const auto baseEntity = this->model.LinkByName(_ecm, this->baseLinkName);
    if (baseEntity == gz::sim::kNullEntity)
    {
      gzerr << "[VirtualSheetActuator] Base link not found: ["
            << this->baseLinkName << "].\n";
      return;
    }
    this->baseLink = gz::sim::Link(baseEntity);
    this->baseLink.EnableVelocityChecks(_ecm, true);

    if (!this->node.Subscribe(
          this->topic, &VirtualSheetActuatorSystem::OnAllowanceCommand, this))
    {
      gzerr << "[VirtualSheetActuator] Failed to subscribe to ["
            << this->topic << "].\n";
      return;
    }

    this->configured = this->ResolveJoint(_ecm);
    gzmsg << "[VirtualSheetActuator] Ready: topic=" << this->topic
          << ", maximum_allowance_rad=" << this->maximumAllowanceRad
          << ", initial_allowance_rad="
          << this->commandedAllowanceRad.load()
          << ", linear_stiffness_nm_per_rad="
          << this->linearStiffnessNmPerRad
          << ", quadratic_stiffness_nm_per_rad2="
          << this->quadraticStiffnessNmPerRad2
          << ", damping_nm_per_rad_s=" << this->dampingNmPerRadS
          << ", maximum_torque_nm=" << this->maximumTorqueNm
          << ", containment_limit_rad=" << this->containmentLimitRad
          << ", take_up_rate_rad_s=" << this->takeUpRateRadS
          << ", physics_rate_telemetry=" << std::boolalpha
          << this->physicsRateTelemetry << std::endl;
  }

  void PreUpdate(
      const gz::sim::UpdateInfo &_info,
      gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused)
      return;
    if (!this->configured)
      this->configured = this->ResolveJoint(_ecm);
    if (!this->configured)
      return;

    const auto positions = this->joint.Position(_ecm);
    const auto velocities = this->joint.Velocity(_ecm);
    if (!positions || positions->empty() ||
        !velocities || velocities->empty())
    {
      return;
    }

    const double rawAllowanceRad = this->commandedAllowanceRad.load();
    const double dtS = std::chrono::duration<double>(_info.dt).count();
    const auto allowanceState = UpdateVirtualSheetAllowance(
      this->effectiveAllowanceRad, rawAllowanceRad, positions->front(),
      this->maximumAllowanceRad, this->containmentLimitRad,
      this->takeUpRateRadS, dtS);
    this->effectiveAllowanceRad = allowanceState.effectiveAllowanceRad;
    const auto result = ComputeVirtualSheetActuator(
      positions->front(), velocities->front(), this->effectiveAllowanceRad,
      this->maximumAllowanceRad, this->linearStiffnessNmPerRad,
      this->quadraticStiffnessNmPerRad2, this->dampingNmPerRadS,
      this->maximumTorqueNm);

    auto forceComponent =
      _ecm.Component<gz::sim::components::JointForceCmd>(this->joint.Entity());
    if (forceComponent == nullptr)
    {
      _ecm.CreateComponent(
        this->joint.Entity(),
        gz::sim::components::JointForceCmd({result.torqueNm}));
    }
    else
    {
      *forceComponent =
        gz::sim::components::JointForceCmd({result.torqueNm});
    }

    if (this->physicsRateTelemetry)
      this->LogPhysics(
        _info, _ecm, allowanceState, result,
        positions->front(), velocities->front());
  }

private:
  void OnAllowanceCommand(const gz::msgs::Double &_message)
  {
    if (!std::isfinite(_message.data()))
    {
      gzwarn << "[VirtualSheetActuator] Ignoring non-finite allowance command.\n";
      return;
    }
    this->commandedAllowanceRad.store(_message.data());
    this->commandUpdateCount.fetch_add(1);
  }

  bool ResolveJoint(gz::sim::EntityComponentManager &_ecm)
  {
    if (this->joint.Entity() != gz::sim::kNullEntity)
      return true;

    const auto jointEntity = this->model.JointByName(_ecm, this->jointName);
    if (jointEntity == gz::sim::kNullEntity)
      return false;

    this->joint = gz::sim::Joint(jointEntity);
    this->joint.EnablePositionCheck(_ecm, true);
    this->joint.EnableVelocityCheck(_ecm, true);
    return true;
  }

  void LogPhysics(
      const gz::sim::UpdateInfo &_info,
      const gz::sim::EntityComponentManager &_ecm,
      const VirtualSheetAllowanceStateResult &_allowanceState,
      const VirtualSheetActuatorResult &_result,
      const double _positionRad,
      const double _velocityRadS) const
  {
    const auto baseLinearVelocity = this->baseLink.WorldLinearVelocity(_ecm);
    const auto baseAngularVelocity = this->baseLink.WorldAngularVelocity(_ecm);
    if (!baseLinearVelocity || !baseAngularVelocity)
      return;

    const double simTimeS =
      std::chrono::duration<double>(_info.simTime).count();
    std::ostringstream line;
    line << std::setprecision(12)
         << "[VirtualSheetActuator][physics]"
         << " simTimeS=" << simTimeS
         << " commandUpdateCount=" << this->commandUpdateCount.load()
         << " rawAllowanceRad=" << _allowanceState.rawAllowanceRad
         << " effectiveAllowanceRad="
         << _allowanceState.effectiveAllowanceRad
         << " allowanceLagRad=" << _allowanceState.lagRad
         << " allowanceSafeFloorRad=" << _allowanceState.safeFloorRad
         << " allowanceSheetOutRelease="
         << (_allowanceState.sheetOutRelease ? "true" : "false")
         << " allowanceTightening="
         << (_allowanceState.tightening ? "true" : "false")
         << " allowanceRad=" << _result.allowanceRad
         << " boomAngleRad=" << _positionRad
         << " boomVelocityRadS=" << _velocityRadS
         << " rawPenetrationRad="
         << std::max(
              std::abs(_positionRad) - _allowanceState.rawAllowanceRad, 0.0)
         << " effectivePenetrationRad=" << _result.penetrationRad
         << " penetrationRad=" << _result.penetrationRad
         << " outwardVelocityRadS=" << _result.outwardVelocityRadS
         << " springTorqueNm=" << _result.springTorqueNm
         << " dampingTorqueNm=" << _result.dampingTorqueNm
         << " requestedTorqueNm=" << _result.requestedTorqueNm
         << " appliedTorqueNm=" << _result.torqueNm
         << " sheetTaut=" << (_result.active ? "true" : "false")
         << " torqueCapped=" << (_result.capped ? "true" : "false")
         << " physicalContainmentViolation="
         << (_allowanceState.containmentViolation ? "true" : "false")
         << " baseLinearVelocityWorld=" << *baseLinearVelocity
         << " baseAngularVelocityWorld=" << *baseAngularVelocity;
    gzerr << line.str() << std::endl;
  }

private:
  gz::sim::Model model{gz::sim::kNullEntity};
  gz::sim::Joint joint{gz::sim::kNullEntity};
  gz::sim::Link baseLink{gz::sim::kNullEntity};
  gz::transport::Node node;
  std::string jointName{"sail_joint"};
  std::string baseLinkName{"base_link"};
  std::string topic{"/model/sailboat/joint/sail_joint/cmd_pos"};
  std::atomic<double> commandedAllowanceRad{0.7854};
  std::atomic<std::uint64_t> commandUpdateCount{0};
  double effectiveAllowanceRad{0.7854};
  double maximumAllowanceRad{0.7854};
  double linearStiffnessNmPerRad{300.0};
  double quadraticStiffnessNmPerRad2{460.0};
  double dampingNmPerRadS{44.0};
  double maximumTorqueNm{260.0};
  double containmentLimitRad{0.2617993877991494};
  double takeUpRateRadS{0.5235987755982988};
  bool physicsRateTelemetry{true};
  bool configured{false};
};
}  // namespace ioes::sim

GZ_ADD_PLUGIN(
  ioes::sim::VirtualSheetActuatorSystem,
  gz::sim::System,
  ioes::sim::VirtualSheetActuatorSystem::ISystemConfigure,
  ioes::sim::VirtualSheetActuatorSystem::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(
  ioes::sim::VirtualSheetActuatorSystem,
  "ioes::sim::VirtualSheetActuatorSystem")
