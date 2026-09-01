#include <gz/common/Console.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>
#include <gz/msgs/boolean.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/AngularVelocityCmd.hh>
#include <gz/sim/components/LinearVelocityCmd.hh>
#include <gz/transport/Node.hh>
#include <sdf/Element.hh>

#include <chrono>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>

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

/// Hold a model at its configured spawn pose while flight-controller sensors
/// initialize. A Boolean message on holdTopic controls the hold: true holds,
/// false releases. Once released, this system no longer modifies the model.
class StartupStateHoldSystem:
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
      gzerr << "[StartupStateHoldSystem] Plugin must be attached to a model.\n";
      return;
    }

    this->initialWorldPose = gz::sim::worldPose(_entity, _ecm);
    const std::string modelName = this->model.Name(_ecm);
    this->holdTopic = GetParam<std::string>(
      _sdf, "hold_topic", "/model/" + modelName + "/startup_hold");
    this->requestedHeld = GetParam<bool>(_sdf, "initially_held", true);
    this->appliedHeld = this->requestedHeld;
    this->autoReleaseTimeoutS = GetParam<double>(
      _sdf, "auto_release_timeout_s", 120.0);

    if (!this->node.Subscribe(
          this->holdTopic, &StartupStateHoldSystem::OnHoldMsg, this))
    {
      gzerr << "[StartupStateHoldSystem] Failed to subscribe to ["
            << this->holdTopic << "]. Disabling the startup hold.\n";
      this->requestedHeld = false;
      this->appliedHeld = false;
      return;
    }

    gzmsg << "[StartupStateHoldSystem] Loaded for model [" << modelName
          << "], topic=[" << this->holdTopic << "], initially_held="
          << std::boolalpha << this->appliedHeld
          << ", auto_release_timeout_s="
          << this->autoReleaseTimeoutS << std::endl;
  }

  void PreUpdate(
      const gz::sim::UpdateInfo &_info,
      gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused || !this->model.Valid(_ecm))
      return;

    const double simTimeS =
      std::chrono::duration<double>(_info.simTime).count();
    bool requestedHeld = false;
    {
      std::lock_guard<std::mutex> lock(this->stateMutex);
      requestedHeld = this->requestedHeld;
    }

    bool autoReleased = false;
    if (this->appliedHeld && this->holdStartSimTimeS >= 0.0 &&
        this->autoReleaseTimeoutS > 0.0 &&
        simTimeS - this->holdStartSimTimeS >= this->autoReleaseTimeoutS)
    {
      std::lock_guard<std::mutex> lock(this->stateMutex);
      this->requestedHeld = false;
      requestedHeld = false;
      autoReleased = true;
    }

    if (autoReleased)
    {
      gzwarn << "[StartupStateHoldSystem] Auto-released startup hold after "
             << this->autoReleaseTimeoutS << " simulation seconds.\n";
    }

    if (requestedHeld != this->appliedHeld)
    {
      const bool previouslyHeld = this->appliedHeld;
      this->appliedHeld = requestedHeld;
      this->holdStartSimTimeS = this->appliedHeld ? simTimeS : -1.0;

      gzwarn << "[StartupStateHoldSystem] internal held-state transition: "
             << std::boolalpha << previouslyHeld << " -> "
             << this->appliedHeld << ", source="
             << (autoReleased ? "auto_timeout" : "transport") << std::endl;

      if (previouslyHeld && !this->appliedHeld)
        this->CleanupVelocityCommands(_ecm);
    }

    if (!this->appliedHeld)
      return;

    if (this->holdStartSimTimeS < 0.0)
      this->holdStartSimTimeS = simTimeS;

    // Override pre-mission wave / hydrodynamic drift while leaving all sensors
    // and the simulator clock active for EKF and pre-arm initialization.
    this->model.SetWorldPoseCmd(_ecm, this->initialWorldPose);
    for (const auto linkEntity : this->model.Links(_ecm))
    {
      gz::sim::Link link(linkEntity);
      link.SetLinearVelocity(_ecm, gz::math::Vector3d::Zero);
      link.SetAngularVelocity(_ecm, gz::math::Vector3d::Zero);
      this->velocityCommandLinks.insert(linkEntity);
    }
  }

private:
  void CleanupVelocityCommands(gz::sim::EntityComponentManager &_ecm)
  {
    std::size_t linearRemoved = 0u;
    std::size_t angularRemoved = 0u;
    for (const auto linkEntity : this->velocityCommandLinks)
    {
      linearRemoved += _ecm.RemoveComponent<
        gz::sim::components::LinearVelocityCmd>(linkEntity) ? 1u : 0u;
      angularRemoved += _ecm.RemoveComponent<
        gz::sim::components::AngularVelocityCmd>(linkEntity) ? 1u : 0u;
    }
    this->velocityCommandLinks.clear();

    gzwarn << "[StartupStateHoldSystem] velocity command cleanup: "
           << "linear_removed=" << linearRemoved
           << ", angular_removed=" << angularRemoved << std::endl;
  }

  void OnHoldMsg(const gz::msgs::Boolean &_msg)
  {
    bool shouldLog = false;
    {
      std::lock_guard<std::mutex> lock(this->stateMutex);
      this->requestedHeld = _msg.data();
      shouldLog = !this->receivedHoldCommand ||
        this->lastLoggedRequestedHeld != this->requestedHeld;
      this->receivedHoldCommand = true;
      this->lastLoggedRequestedHeld = this->requestedHeld;
    }

    if (shouldLog)
    {
      gzwarn << "[StartupStateHoldSystem] hold command received: "
             << "requested_held=" << std::boolalpha << _msg.data()
             << std::endl;
    }
  }

private:
  gz::sim::Model model{gz::sim::kNullEntity};
  gz::math::Pose3d initialWorldPose;
  gz::transport::Node node;
  std::mutex stateMutex;
  std::string holdTopic;
  bool requestedHeld{true};
  bool appliedHeld{true};
  bool receivedHoldCommand{false};
  bool lastLoggedRequestedHeld{true};
  double holdStartSimTimeS{-1.0};
  double autoReleaseTimeoutS{120.0};
  std::unordered_set<gz::sim::Entity> velocityCommandLinks;
};
}  // namespace ioes::sim

GZ_ADD_PLUGIN(
  ioes::sim::StartupStateHoldSystem,
  gz::sim::System,
  ioes::sim::StartupStateHoldSystem::ISystemConfigure,
  ioes::sim::StartupStateHoldSystem::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(
  ioes::sim::StartupStateHoldSystem,
  "ioes::sim::StartupStateHoldSystem")
