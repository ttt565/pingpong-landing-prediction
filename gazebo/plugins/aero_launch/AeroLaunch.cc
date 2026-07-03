#include "AeroLaunch.hh"

#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/AngularVelocityCmd.hh>
#include <gz/sim/components/LinearVelocityCmd.hh>

using namespace ttsim;

static gz::math::Vector3d ParseVec(const std::shared_ptr<const sdf::Element> &_sdf,
                                   const std::string &_name,
                                   const gz::math::Vector3d &_def)
{
  if (_sdf->HasElement(_name))
    return _sdf->Get<gz::math::Vector3d>(_name);
  return _def;
}

void AeroLaunch::Configure(const gz::sim::Entity &_entity,
                           const std::shared_ptr<const sdf::Element> &_sdf,
                           gz::sim::EntityComponentManager &_ecm,
                           gz::sim::EventManager &)
{
  gz::sim::Model model(_entity);
  if (_sdf->HasElement("link_name"))
    this->linkName = _sdf->Get<std::string>("link_name");
  this->mass        = _sdf->Get<double>("mass", this->mass).first;
  this->dragCoeff   = _sdf->Get<double>("drag_coeff", this->dragCoeff).first;
  this->magnusCoeff = _sdf->Get<double>("magnus_coeff", this->magnusCoeff).first;
  this->initLinear  = ParseVec(_sdf, "init_linear", this->initLinear);
  this->initAngular = ParseVec(_sdf, "init_angular", this->initAngular);

  this->linkEntity = model.LinkByName(_ecm, this->linkName);
  if (this->linkEntity == gz::sim::kNullEntity)
  {
    gzerr << "[AeroLaunch] link '" << this->linkName << "' not found\n";
    return;
  }
  gz::sim::Link link(this->linkEntity);
  link.EnableVelocityChecks(_ecm, true);  // needed to read world velocities
}

void AeroLaunch::PreUpdate(const gz::sim::UpdateInfo &_info,
                           gz::sim::EntityComponentManager &_ecm)
{
  if (_info.paused || this->linkEntity == gz::sim::kNullEntity)
    return;

  gz::sim::Link link(this->linkEntity);

  // (1) one-shot launch: impose initial linear + angular velocity
  if (!this->launched)
  {
    link.SetLinearVelocity(_ecm, this->initLinear);
    link.SetAngularVelocity(_ecm, this->initAngular);
    this->launched = true;
    return;  // let physics integrate one step before we start pushing
  }

  // (1b) SetLinearVelocity/SetAngularVelocity create *VelocityCmd components.
  // The physics system re-applies those commands EVERY step and zeroes their
  // value after each step (it never removes them) — leaving them in the ECM
  // pins the ball's velocity to zero forever. Remove them once, the step
  // after launch, so the ball flies ballistically from here on.
  if (!this->cmdsCleared)
  {
    _ecm.RemoveComponent<gz::sim::components::LinearVelocityCmd>(this->linkEntity);
    _ecm.RemoveComponent<gz::sim::components::AngularVelocityCmd>(this->linkEntity);
    this->cmdsCleared = true;
  }

  // (2) aerodynamic force every step: F = mass * (a_drag + a_magnus)
  auto vOpt = link.WorldLinearVelocity(_ecm);
  auto wOpt = link.WorldAngularVelocity(_ecm);
  if (!vOpt.has_value() || !wOpt.has_value())
    return;

  const gz::math::Vector3d v = *vOpt;
  const gz::math::Vector3d w = *wOpt;
  const gz::math::Vector3d aDrag   = -this->dragCoeff * v.Length() * v;
  const gz::math::Vector3d aMagnus =  this->magnusCoeff * w.Cross(v);
  link.AddWorldForce(_ecm, this->mass * (aDrag + aMagnus));
}

GZ_ADD_PLUGIN(ttsim::AeroLaunch,
              gz::sim::System,
              ttsim::AeroLaunch::ISystemConfigure,
              ttsim::AeroLaunch::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(ttsim::AeroLaunch, "ttsim::AeroLaunch")
