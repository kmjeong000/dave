#include "dave_gz_model_plugins/LiftDragModel.hh"

#include <gz/common/Console.hh>
#include <gz/math/Helpers.hh>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/double.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/transport/Node.hh>
#include <sdf/Element.hh>

#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
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

struct RudderWrenchDiagnostics
{
  double jointAngleRad{0.0};
  gz::math::Vector3d cpWorld{0, 0, 0};
  gz::math::Vector3d leverFromBaseWorld{0, 0, 0};
  gz::math::Vector3d momentAboutBaseWorld{0, 0, 0};
  gz::math::Vector3d yawAxisWorld{0, 0, 0};
  double yawMomentNm{0.0};
};

RudderWrenchDiagnostics ComputeRudderWrenchDiagnostics(
    const gz::math::Pose3d &_foilPose,
    const gz::math::Pose3d &_basePose,
    const gz::math::Vector3d &_cpLink,
    const gz::math::Vector3d &_yawAxisBody,
    const gz::math::Vector3d &_forceWorld)
{
  RudderWrenchDiagnostics diagnostics;
  const auto relativeRotation =
    _basePose.Rot().Inverse() * _foilPose.Rot();
  const double relativeYaw = relativeRotation.Yaw();
  diagnostics.jointAngleRad = std::atan2(
    std::sin(relativeYaw), std::cos(relativeYaw));
  diagnostics.cpWorld =
    _foilPose.Pos() + _foilPose.Rot().RotateVector(_cpLink);
  diagnostics.leverFromBaseWorld =
    diagnostics.cpWorld - _basePose.Pos();
  diagnostics.momentAboutBaseWorld =
    diagnostics.leverFromBaseWorld.Cross(_forceWorld);
  diagnostics.yawAxisWorld =
    _basePose.Rot().RotateVector(_yawAxisBody);
  diagnostics.yawMomentNm =
    diagnostics.momentAboutBaseWorld.Dot(diagnostics.yawAxisWorld);
  return diagnostics;
}
}

// Diagnostic-only counterpart of FoilLiftDragSystem.  It is selected only in
// the runner's trial-local SDF and preserves the same LiftDragModel force math.
// The configured test flow is held at zero until a transport command is
// consumed by PreUpdate immediately before the fixed-rudder measurement.
class RudderDiagnosticLoadSystem:
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
    this->model = gz::sim::Model(_entity);
    if (!this->model.Valid(_ecm))
    {
      gzerr << "[RudderDiagnosticLoadSystem] Plugin must be attached to a model.\n";
      return;
    }

    this->linkName = GetParam<std::string>(_sdf, "link_name", "");
    if (this->linkName.empty())
      this->linkName = GetParam<std::string>(_sdf, "foil_link", "");
    if (this->linkName.empty())
      this->linkName = GetParam<std::string>(_sdf, "daggerboard_link", "");
    const auto linkEntity = this->model.LinkByName(_ecm, this->linkName);
    if (this->linkName.empty() || linkEntity == gz::sim::kNullEntity)
    {
      gzerr << "[RudderDiagnosticLoadSystem] Rudder link ["
            << this->linkName << "] was not found.\n";
      return;
    }
    this->link = gz::sim::Link(linkEntity);
    this->link.EnableVelocityChecks(_ecm, true);

    this->baseLinkName = GetParam<std::string>(_sdf, "base_link", "");
    const auto baseLinkEntity =
      this->model.LinkByName(_ecm, this->baseLinkName);
    if (this->baseLinkName.empty() ||
        baseLinkEntity == gz::sim::kNullEntity)
    {
      gzerr << "[RudderDiagnosticLoadSystem] Base link ["
            << this->baseLinkName << "] was not found.\n";
      return;
    }
    this->baseLink = gz::sim::Link(baseLinkEntity);

    this->yawAxisBody =
      GetParam<gz::math::Vector3d>(_sdf, "yaw_axis", {0, 0, 1});
    const double yawAxisLength = this->yawAxisBody.Length();
    if (yawAxisLength <= 1e-12)
      this->yawAxisBody.Set(0, 0, 1);
    else
      this->yawAxisBody /= yawAxisLength;

    LiftDragParams params;
    params.fluidDensity = GetParam<double>(_sdf, "fluid_density", 1000.0);
    params.radialSymmetry = GetParam<bool>(_sdf, "radial_symmetry", true);
    params.forward =
      GetParam<gz::math::Vector3d>(_sdf, "forward", {1, 0, 0});
    params.upward =
      GetParam<gz::math::Vector3d>(_sdf, "upward", {0, 0, 1});
    params.cp = GetParam<gz::math::Vector3d>(_sdf, "cp", {0, 0, 0});
    params.area = GetParam<double>(_sdf, "area", 0.1);
    params.alpha0 = GetParam<double>(_sdf, "a0", 0.0);
    params.alphaStall =
      GetParam<double>(_sdf, "alpha_stall", GZ_PI / 4.0);
    params.cla = GetParam<double>(_sdf, "cla", 4.0 / GZ_PI);
    params.claStall =
      GetParam<double>(_sdf, "cla_stall", -4.0 / GZ_PI);
    params.cda = GetParam<double>(_sdf, "cda", 0.0);
    params.minSpeed = GetParam<double>(_sdf, "min_speed", 0.01);
    this->modelData.SetParams(params);

    this->targetWaterVelocityWorld =
      GetParam<gz::math::Vector3d>(
        _sdf, "diagnostic_water_velocity", {0, 0, 0});
    this->enableTopic =
      GetParam<std::string>(_sdf, "diagnostic_flow_enable_topic", "");
    this->flowEnabled =
      GetParam<bool>(_sdf, "diagnostic_flow_enabled", false);
    this->requestedFlowEnabled = this->flowEnabled;
    if (this->enableTopic.empty())
    {
      gzerr << "[RudderDiagnosticLoadSystem] Missing "
            << "<diagnostic_flow_enable_topic>. Test flow remains disabled.\n";
    }
    else if (!this->node.Subscribe(
      this->enableTopic,
      &RudderDiagnosticLoadSystem::OnEnableMsg,
      this))
    {
      gzerr << "[RudderDiagnosticLoadSystem] Failed to subscribe to ["
            << this->enableTopic << "]. Test flow remains disabled.\n";
    }

    this->commandTopic = GetParam<std::string>(_sdf, "command_topic", "");
    if (!this->commandTopic.empty())
    {
      this->node.Subscribe(
        this->commandTopic,
        &RudderDiagnosticLoadSystem::OnCommandMsg,
        this);
    }
    this->debug = GetParam<bool>(_sdf, "debug", false);
    this->debugPeriod = GetParam<double>(_sdf, "debug_period", 1.0);

    gzerr << "[RudderDiagnosticLoadSystem] Loaded for link ["
          << this->linkName << "], initial_water_velocity=0 0 0"
          << ", target_water_velocity=" << this->targetWaterVelocityWorld
          << ", initial_enabled=" << (this->flowEnabled ? "true" : "false")
          << "\n";
  }

  void PreUpdate(const gz::sim::UpdateInfo &_info,
                 gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused || this->link.Entity() == gz::sim::kNullEntity)
      return;

    const double simSec = std::chrono::duration<double>(_info.simTime).count();
    const auto waterVelocityWorld = this->CurrentWaterVelocityWorld(simSec);
    const auto poseOpt = this->link.WorldPose(_ecm);
    const auto velOpt =
      this->link.WorldLinearVelocity(_ecm, this->modelData.Params().cp);
    const auto basePoseOpt = this->baseLink.WorldPose(_ecm);
    if (!poseOpt || !velOpt || !basePoseOpt)
      return;

    const gz::math::Vector3d freeStream = waterVelocityWorld - *velOpt;
    const auto result = this->modelData.Compute(freeStream, *poseOpt);
    const auto wrench = ComputeRudderWrenchDiagnostics(
      *poseOpt,
      *basePoseOpt,
      this->modelData.Params().cp,
      this->yawAxisBody,
      result.force);

    if (this->debug && simSec - this->lastDebugTime >= this->debugPeriod)
    {
      this->lastDebugTime = simSec;
      const double commandRad = this->CurrentCommandRad();
      // Keep the established parser-compatible prefix and fields.
      gzerr << "[FoilLiftDragSystem] link=" << this->linkName
            << " simTimeS=" << simSec
            << " Vfluid=" << freeStream
            << " speed=" << result.speed
            << " alpha_deg="
            << (result.alpha * 180.0 / 3.14159265358979323846)
            << " cl=" << result.cl
            << " cd=" << result.cd
            << " lift=" << result.lift
            << " drag=" << result.drag
            << " forceWorld=" << result.force
            << " jointAngleDeg="
            << (wrench.jointAngleRad * 180.0 / 3.14159265358979323846)
            << " commandDeg="
            << (commandRad * 180.0 / 3.14159265358979323846)
            << " cpWorld=" << wrench.cpWorld
            << " leverFromBaseWorld=" << wrench.leverFromBaseWorld
            << " momentAboutBaseWorld=" << wrench.momentAboutBaseWorld
            << " yawAxisWorld=" << wrench.yawAxisWorld
            << " yawMomentNm=" << wrench.yawMomentNm << "\n";
    }

    if (result.force.Length() > 0.0)
    {
      this->link.AddWorldWrench(
        _ecm,
        result.force,
        gz::math::Vector3d::Zero,
        this->modelData.Params().cp);
    }
  }

private:
  void OnEnableMsg(const gz::msgs::Boolean &_msg)
  {
    {
      std::lock_guard<std::mutex> lock(this->flowMutex);
      this->requestedFlowEnabled = _msg.data();
    }
    gzerr << "[RudderDiagnosticLoadSystem] flow command received: "
          << "requested_enabled=" << (_msg.data() ? "true" : "false")
          << "\n";
  }

  void OnCommandMsg(const gz::msgs::Double &_msg)
  {
    std::lock_guard<std::mutex> lock(this->commandMutex);
    this->lastCommandRad = _msg.data();
  }

  gz::math::Vector3d CurrentWaterVelocityWorld(double _simSec)
  {
    std::lock_guard<std::mutex> lock(this->flowMutex);
    if (this->flowEnabled != this->requestedFlowEnabled)
    {
      const bool previous = this->flowEnabled;
      this->flowEnabled = this->requestedFlowEnabled;
      gzerr << "[RudderDiagnosticLoadSystem] diagnostic flow transition: "
            << "link=" << this->linkName << ", "
            << (previous ? "true" : "false") << " -> "
            << (this->flowEnabled ? "true" : "false")
            << ", simTimeS=" << _simSec
            << ", active_water_velocity="
            << (this->flowEnabled
                  ? this->targetWaterVelocityWorld
                  : gz::math::Vector3d::Zero)
            << "\n";
    }
    return this->flowEnabled
      ? this->targetWaterVelocityWorld
      : gz::math::Vector3d::Zero;
  }

  double CurrentCommandRad()
  {
    std::lock_guard<std::mutex> lock(this->commandMutex);
    return this->lastCommandRad;
  }

  gz::sim::Model model{gz::sim::kNullEntity};
  gz::sim::Link link{gz::sim::kNullEntity};
  gz::sim::Link baseLink{gz::sim::kNullEntity};
  gz::transport::Node node;
  std::mutex flowMutex;
  std::mutex commandMutex;
  std::string linkName;
  std::string baseLinkName;
  std::string enableTopic;
  std::string commandTopic;
  LiftDragModel modelData;
  gz::math::Vector3d targetWaterVelocityWorld{0, 0, 0};
  gz::math::Vector3d yawAxisBody{0, 0, 1};
  bool flowEnabled{false};
  bool requestedFlowEnabled{false};
  double lastCommandRad{0.0};
  bool debug{false};
  double debugPeriod{1.0};
  double lastDebugTime{-1e9};
};
}  // namespace ioes::sim

GZ_ADD_PLUGIN(
  ioes::sim::RudderDiagnosticLoadSystem,
  gz::sim::System,
  ioes::sim::RudderDiagnosticLoadSystem::ISystemConfigure,
  ioes::sim::RudderDiagnosticLoadSystem::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(
  ioes::sim::RudderDiagnosticLoadSystem,
  "ioes::sim::RudderDiagnosticLoadSystem")
