#include "dave_gz_model_plugins/LiftDragModel.hh"

#include <gz/common/Console.hh>
#include <gz/math/Helpers.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/LinearVelocity.hh>
#include <gz/sim/components/LinearVelocitySeed.hh>
#include <gz/sim/components/Wind.hh>
#include <sdf/Element.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/wind.pb.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <mutex>

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

double Clamp(double _value, double _min, double _max)
{
  return std::max(_min, std::min(_max, _value));
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
    this->windTopic = GetParam<std::string>(_sdf, "wind_topic", "");
    this->targetForceEnabled = GetParam<bool>(_sdf, "enabled", true);
    this->targetForceScale = std::max(0.0, GetParam<double>(_sdf, "force_scale", 1.0));
    this->maxForceN = std::max(0.0, GetParam<double>(_sdf, "max_force_n", 0.0));
    this->enableRampTimeS = std::max(0.0, GetParam<double>(_sdf, "enable_ramp_time_s", 0.0));
    this->enableTopic = GetParam<std::string>(_sdf, "enable_topic", "");
    if (this->enableTopic.empty())
    {
      const auto modelName = this->model.Name(_ecm);
      this->enableTopic = "/model/" + modelName + "/sail_lift_drag/enable";
    }
    this->windEntity = _ecm.EntityByComponents(gz::sim::components::Wind());
    if (this->windEntity != gz::sim::kNullEntity)
    {
      gzmsg << "[SailLiftDragSystem] Found Gazebo wind entity. "
            << "Using world wind component when available." << std::endl;
    }
    else
    {
      gzwarn << "[SailLiftDragSystem] Gazebo wind entity not found at configure time. "
             << "Will retry during updates." << std::endl;
    }

    if (!this->windTopic.empty())
    {
      const bool subscribed = 
        this->node.Subscribe(this->windTopic,
                            &SailLiftDragSystem::OnWindMsg, 
                            this);
      if (subscribed)
      {
        gzerr << "[SailLiftDragSystem] Subscribed to wind topic ["
              << this->windTopic << "]" << std::endl;
      }
      else
      {
        gzerr << "[SailLiftDragSystem] Failed to subscribe to wind topic ["
              << this->windTopic << "]" << std::endl;
      }
    }
    else
    {
      gzerr << "[SailLiftDragSystem] No <wind_topic>. Using fixed wind_velocity="
            << this->windVelocityWorld << std::endl;
    }

    if (!this->enableTopic.empty())
    {
      const bool subscribed =
        this->node.Subscribe(this->enableTopic,
                            &SailLiftDragSystem::OnEnableMsg,
                            this);
      if (subscribed)
      {
        gzerr << "[SailLiftDragSystem] Subscribed to enable topic ["
              << this->enableTopic << "], initial_enabled="
              << (this->targetForceEnabled ? "true" : "false")
              << ", force_scale=" << this->targetForceScale
              << ", max_force_n=" << this->maxForceN
              << ", enable_ramp_time_s=" << this->enableRampTimeS
              << std::endl;
      }
      else
      {
        gzerr << "[SailLiftDragSystem] Failed to subscribe to enable topic ["
              << this->enableTopic << "]" << std::endl;
      }
    }

    this->debug = GetParam<bool>(_sdf, "debug", false);
    this->debugPeriod = GetParam<double>(_sdf, "debug_period", 1.0);

    gzmsg << "[SailLiftDragSystem] Loaded for link [" << this->linkName
          << "], wind_velocity=" << this->windVelocityWorld
          << ", area=" << p.area << ", rho=" << p.fluidDensity << "\n";
  }

  void OnWindMsg(const gz::msgs::Wind &_msg)
  {
    std::lock_guard<std::mutex> lock(this->windMutex);

    if (_msg.has_linear_velocity())
    {
      const auto &v = _msg.linear_velocity();
      this->windVelocityWorld.Set(v.x(), v.y(), v.z());
      this->windTopicReceived = true;
    }
  }

  void OnEnableMsg(const gz::msgs::Boolean &_msg)
  {
    {
      std::lock_guard<std::mutex> lock(this->controlMutex);
      const bool previous = this->targetForceEnabled;
      this->targetForceEnabled = _msg.data();
      if (!this->targetForceEnabled)
      {
        this->rampStartSimTimeS = -1.0;
        this->lastForceScale = 0.0;
      }
      else if (!previous)
      {
        this->rampStartSimTimeS = -1.0;
        this->lastForceScale = 0.0;
      }
    }
    gzerr << "[SailLiftDragSystem] force_enabled="
          << (_msg.data() ? "true" : "false")
          << ", ramp_time_s=" << this->enableRampTimeS
          << ", target_force_scale=" << this->targetForceScale
          << std::endl;
  }

  double CurrentForceScale(double _simSec)
  {
    std::lock_guard<std::mutex> lock(this->controlMutex);
    if (!this->targetForceEnabled)
    {
      this->lastForceScale = 0.0;
      return 0.0;
    }

    if (this->enableRampTimeS <= 1e-9)
    {
      this->lastForceScale = this->targetForceScale;
      return this->lastForceScale;
    }

    if (this->rampStartSimTimeS < 0.0)
      this->rampStartSimTimeS = _simSec;

    const double rampRatio =
      Clamp((_simSec - this->rampStartSimTimeS) / this->enableRampTimeS, 0.0, 1.0);
    this->lastForceScale = this->targetForceScale * rampRatio;
    return this->lastForceScale;
  }

  gz::math::Vector3d CurrentWindWorld(
      gz::sim::EntityComponentManager &_ecm,
      std::string &_source)
  {
    if (this->windEntity == gz::sim::kNullEntity)
      this->windEntity = _ecm.EntityByComponents(gz::sim::components::Wind());

    if (this->windEntity != gz::sim::kNullEntity)
    {
      auto windVel =
        _ecm.Component<gz::sim::components::WorldLinearVelocity>(this->windEntity);
      if (windVel != nullptr)
      {
        _source = "world_component";
        return windVel->Data();
      }

      auto windSeed =
        _ecm.Component<gz::sim::components::WorldLinearVelocitySeed>(
          this->windEntity);
      if (windSeed != nullptr)
      {
        _source = "world_seed";
        return windSeed->Data();
      }
    }

    std::lock_guard<std::mutex> lock(this->windMutex);
    _source = this->windTopicReceived ? "wind_topic" : "sdf_fallback";
    return this->windVelocityWorld;
  }

  void PreUpdate(const gz::sim::UpdateInfo &_info,
                 gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused || this->link.Entity() == gz::sim::kNullEntity)
      return;

    const double simSec = std::chrono::duration<double>(_info.simTime).count();
    const double forceScale = this->CurrentForceScale(simSec);
    if (forceScale <= 1e-9)
      return;

    const auto poseOpt = this->link.WorldPose(_ecm);
    const auto velOpt = this->link.WorldLinearVelocity(_ecm, this->modelData.Params().cp);
    if (!poseOpt || !velOpt)
      return;

    // Equivalent to asv_sim SailPlugin:
    // free stream = wind velocity at sail - sail CP velocity.
    std::string windSource;
    const gz::math::Vector3d windWorld =
      this->CurrentWindWorld(_ecm, windSource);
    const gz::math::Vector3d freeStream = windWorld - *velOpt;
    auto result = this->modelData.Compute(freeStream, *poseOpt);

    if (result.force.Length() <= 0.0)
      return;

    const double rawForceN = result.force.Length();
    result.lift = forceScale * result.lift;
    result.drag = forceScale * result.drag;
    result.force = forceScale * result.force;

    double forceLimitScale = 1.0;
    const double scaledForceN = result.force.Length();
    if (this->maxForceN > 0.0 && scaledForceN > this->maxForceN)
    {
      forceLimitScale = this->maxForceN / scaledForceN;
      result.lift = forceLimitScale * result.lift;
      result.drag = forceLimitScale * result.drag;
      result.force = forceLimitScale * result.force;
    }

    // Apply force at the center of pressure. The offset is in the link frame.
    this->link.AddWorldWrench(
      _ecm, result.force, gz::math::Vector3d::Zero, this->modelData.Params().cp);

    if (this->debug)
    {
      if (simSec - this->lastDebugTime >= this->debugPeriod)
      {
        this->lastDebugTime = simSec;
        gzmsg << "[SailLiftDragSystem] link=" << this->linkName
              << " windSource=" << windSource
              << " forceScale=" << forceScale
              << " forceLimitScale=" << forceLimitScale
              << " rawForceN=" << rawForceN
              << " windWorld=" << windWorld
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
  gz::sim::Entity windEntity{gz::sim::kNullEntity};
  gz::transport::Node node;
  std::mutex windMutex;
  mutable std::mutex controlMutex;
  std::string windTopic;
  std::string enableTopic;
  std::string linkName;
  LiftDragModel modelData;
  gz::math::Vector3d windVelocityWorld{0, 0, 0};
  bool windTopicReceived{false};
  bool targetForceEnabled{true};
  double targetForceScale{1.0};
  double maxForceN{0.0};
  double enableRampTimeS{0.0};
  double rampStartSimTimeS{-1.0};
  double lastForceScale{0.0};
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
