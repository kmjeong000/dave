#include "dave_gz_model_plugins/LiftDragModel.hh"

#include <gz/common/Console.hh>
#include <gz/math/Helpers.hh>
#include <gz/plugin/Register.hh>
#include <gz/msgs/double.pb.h>
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

struct FoilWrenchDiagnostics
{
  double jointAngleRad{0.0};
  gz::math::Vector3d cpWorld{0, 0, 0};
  gz::math::Vector3d leverFromBaseWorld{0, 0, 0};
  gz::math::Vector3d momentAboutBaseWorld{0, 0, 0};
  gz::math::Vector3d yawAxisWorld{0, 0, 0};
  double yawMomentNm{0.0};
};

FoilWrenchDiagnostics ComputeFoilWrenchDiagnostics(
    const gz::math::Pose3d &_foilPose,
    const gz::math::Pose3d &_basePose,
    const gz::math::Vector3d &_cpLink,
    const gz::math::Vector3d &_yawAxisBody,
    const gz::math::Vector3d &_forceWorld)
{
  FoilWrenchDiagnostics diagnostics;
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

class FoilLiftDragSystem:
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
    gzerr << "[FoilLiftDragSystem] Configure called" << std::endl;
    this->model = gz::sim::Model(_entity);
    if (!this->model.Valid(_ecm))
    {
      gzerr << "[FoilLiftDragSystem] Plugin must be attached to a model.\n";
      return;
    }

    this->linkName = GetParam<std::string>(_sdf, "link_name", "");
    if (this->linkName.empty())
      this->linkName = GetParam<std::string>(_sdf, "foil_link", "");
    if (this->linkName.empty())
      this->linkName = GetParam<std::string>(_sdf, "daggerboard_link", "");

    gzerr << "[FoilLiftDragSystem] daggerboard_link = " << this->linkName << std::endl;

    if (this->linkName.empty())
    {
      gzerr << "[FoilLiftDragSystem] Missing <link_name>, <foil_link>, or <daggerboard_link>.\n";
      return;
    }

    const gz::sim::Entity linkEntity = this->model.LinkByName(_ecm, this->linkName);
    if (linkEntity == gz::sim::kNullEntity)
    {
      gzerr << "[FoilLiftDragSystem] Link [" << this->linkName
            << "] not found. No foil forces will be generated.\n";
      return;
    }

    this->link = gz::sim::Link(linkEntity);
    this->link.EnableVelocityChecks(_ecm, true);

    // Optional reference-link diagnostics are enabled only for foils whose SDF
    // supplies <base_link>.  The daggerboard therefore keeps its old behaviour,
    // while the rudder can report its actual joint angle and hull yaw moment.
    this->baseLinkName = GetParam<std::string>(_sdf, "base_link", "");
    if (!this->baseLinkName.empty())
    {
      const gz::sim::Entity baseLinkEntity =
        this->model.LinkByName(_ecm, this->baseLinkName);
      if (baseLinkEntity == gz::sim::kNullEntity)
      {
        gzwarn << "[FoilLiftDragSystem] Base link [" << this->baseLinkName
               << "] not found. Foil force application will continue, but "
               << "joint-angle and yaw-moment diagnostics will be unavailable.\n";
      }
      else
      {
        this->baseLink = gz::sim::Link(baseLinkEntity);
      }
    }

    this->yawAxisBody =
      GetParam<gz::math::Vector3d>(_sdf, "yaw_axis", {0, 0, 1});
    const double yawAxisLength = this->yawAxisBody.Length();
    if (yawAxisLength <= 1e-12)
    {
      gzwarn << "[FoilLiftDragSystem] Invalid zero <yaw_axis>; using model +Z.\n";
      this->yawAxisBody.Set(0, 0, 1);
    }
    else
    {
      this->yawAxisBody /= yawAxisLength;
    }

    LiftDragParams p;
    p.fluidDensity = GetParam<double>(_sdf, "fluid_density", 1000.0);
    p.radialSymmetry = GetParam<bool>(_sdf, "radial_symmetry", true);
    p.forward = GetParam<gz::math::Vector3d>(_sdf, "forward", {1, 0, 0});
    p.upward = GetParam<gz::math::Vector3d>(_sdf, "upward", {0, 0, 1});
    p.cp = GetParam<gz::math::Vector3d>(_sdf, "cp", {0, 0, 0});
    p.area = GetParam<double>(_sdf, "area", 0.1);
    p.alpha0 = GetParam<double>(_sdf, "a0", 0.0);
    p.alphaStall = GetParam<double>(_sdf, "alpha_stall", GZ_PI / 4.0);
    p.cla = GetParam<double>(_sdf, "cla", 4.0 / GZ_PI);
    p.claStall = GetParam<double>(_sdf, "cla_stall", -4.0 / GZ_PI);
    p.cda = GetParam<double>(_sdf, "cda", 0.0);
    p.minSpeed = GetParam<double>(_sdf, "min_speed", 0.01);

    this->modelData.SetParams(p);
    this->waterVelocityWorld =
      GetParam<gz::math::Vector3d>(_sdf, "water_velocity", {0, 0, 0});
    this->commandTopic = GetParam<std::string>(_sdf, "command_topic", "");
    if (!this->commandTopic.empty())
    {
      const bool subscribed = this->node.Subscribe(
        this->commandTopic, &FoilLiftDragSystem::OnCommandMsg, this);
      if (subscribed)
      {
        gzerr << "[FoilLiftDragSystem] Subscribed to command topic ["
              << this->commandTopic << "]\n";
      }
      else
      {
        gzwarn << "[FoilLiftDragSystem] Failed to subscribe to command topic ["
               << this->commandTopic << "]. Command diagnostics will remain zero.\n";
      }
    }
    this->debug = GetParam<bool>(_sdf, "debug", false);
    this->debugPeriod = GetParam<double>(_sdf, "debug_period", 1.0);

    gzmsg << "[FoilLiftDragSystem] Loaded for link [" << this->linkName
          << "], water_velocity=" << this->waterVelocityWorld
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

    // Equivalent to asv_sim FoilPlugin:
    // free stream = water velocity - foil CP velocity. For still water, this is -velCp.
    const gz::math::Vector3d freeStream = this->waterVelocityWorld - *velOpt;
    auto result = this->modelData.Compute(freeStream, *poseOpt);

    if (this->debug)
    {
      const double simSec = std::chrono::duration<double>(_info.simTime).count();
      if (simSec - this->lastDebugTime >= this->debugPeriod)
      {
        this->lastDebugTime = simSec;
        bool wrenchDiagnosticsValid = false;
        FoilWrenchDiagnostics wrenchDiagnostics;
        if (this->baseLink.Entity() != gz::sim::kNullEntity)
        {
          const auto basePoseOpt = this->baseLink.WorldPose(_ecm);
          if (basePoseOpt)
          {
            wrenchDiagnostics = ComputeFoilWrenchDiagnostics(
              *poseOpt,
              *basePoseOpt,
              this->modelData.Params().cp,
              this->yawAxisBody,
              result.force);
            wrenchDiagnosticsValid = true;
          }
        }

        // Use gzerr so the rate-limited diagnostic remains visible with the
        // project's normal Gazebo verbose=0 launch setting.
        const double commandRad = this->CurrentCommandRad();
        gzerr << "[FoilLiftDragSystem] link=" << this->linkName
              << " simTimeS=" << simSec
              << " Vfluid=" << freeStream
              << " speed=" << result.speed
              << " alpha_deg=" << (result.alpha * 180.0 / 3.14159265358979323846)
              << " cl=" << result.cl
              << " cd=" << result.cd
              << " lift=" << result.lift
              << " drag=" << result.drag
              << " forceWorld=" << result.force;
        if (wrenchDiagnosticsValid)
        {
          gzerr << " jointAngleDeg="
                << (wrenchDiagnostics.jointAngleRad * 180.0 /
                    3.14159265358979323846)
                << " commandDeg="
                << (commandRad * 180.0 / 3.14159265358979323846)
                << " cpWorld=" << wrenchDiagnostics.cpWorld
                << " leverFromBaseWorld="
                << wrenchDiagnostics.leverFromBaseWorld
                << " momentAboutBaseWorld="
                << wrenchDiagnostics.momentAboutBaseWorld
                << " yawAxisWorld=" << wrenchDiagnostics.yawAxisWorld
                << " yawMomentNm=" << wrenchDiagnostics.yawMomentNm;
        }
        else
        {
          gzerr << " wrenchDiagnosticsValid=false";
        }
        gzerr << "\n";
      }
    }

    if (result.force.Length() <= 0.0)
      return;

    this->link.AddWorldWrench(
      _ecm, result.force, gz::math::Vector3d::Zero, this->modelData.Params().cp);
  }

  void OnCommandMsg(const gz::msgs::Double &_msg)
  {
    std::lock_guard<std::mutex> lock(this->commandMutex);
    this->lastCommandRad = _msg.data();
  }

  double CurrentCommandRad()
  {
    std::lock_guard<std::mutex> lock(this->commandMutex);
    return this->lastCommandRad;
  }

private:
  gz::sim::Model model{gz::sim::kNullEntity};
  gz::sim::Link link{gz::sim::kNullEntity};
  gz::sim::Link baseLink{gz::sim::kNullEntity};
  gz::transport::Node node;
  std::mutex commandMutex;
  std::string linkName;
  std::string baseLinkName;
  std::string commandTopic;
  LiftDragModel modelData;
  gz::math::Vector3d waterVelocityWorld{0, 0, 0};
  gz::math::Vector3d yawAxisBody{0, 0, 1};
  double lastCommandRad{0.0};
  bool debug{false};
  double debugPeriod{1.0};
  double lastDebugTime{-1e9};
};
}  // namespace ioes::sim

GZ_ADD_PLUGIN(
  ioes::sim::FoilLiftDragSystem,
  gz::sim::System,
  ioes::sim::FoilLiftDragSystem::ISystemConfigure,
  ioes::sim::FoilLiftDragSystem::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(
  ioes::sim::FoilLiftDragSystem,
  "ioes::sim::FoilLiftDragSystem")
