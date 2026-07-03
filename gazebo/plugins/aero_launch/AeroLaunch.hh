#ifndef TTSIM_AEROLAUNCH_HH_
#define TTSIM_AEROLAUNCH_HH_

#include <gz/sim/System.hh>
#include <gz/math/Vector3.hh>

namespace ttsim
{
/// \brief gz-sim System plugin that (1) launches a ball link with a prescribed
/// initial linear + angular velocity on the first update, and (2) applies
/// aerodynamic drag and Magnus force every step, using the SAME lumped model as
/// the analytical sim:
///     a_drag   = -drag_coeff   * |v| * v
///     a_magnus =  magnus_coeff * (omega x v)
/// Force = mass * a, added at the link's center of mass.
///
/// SDF parameters (all optional, defaults match ttsim/physics.py):
///   <link_name>      name of the ball link            (default "ball_link")
///   <mass>           kg                                (default 0.0027)
///   <drag_coeff>     1/m                               (default 0.1117)
///   <magnus_coeff>   s/rad-ish lumped                  (default 0.0016)
///   <init_linear>    "vx vy vz" m/s                    (default "6 0 0.9")
///   <init_angular>   "wx wy wz" rad/s (topspin = +y)   (default "0 400 0")
class AeroLaunch
    : public gz::sim::System,
      public gz::sim::ISystemConfigure,
      public gz::sim::ISystemPreUpdate
{
  public: void Configure(const gz::sim::Entity &_entity,
                         const std::shared_ptr<const sdf::Element> &_sdf,
                         gz::sim::EntityComponentManager &_ecm,
                         gz::sim::EventManager &_eventMgr) override;

  public: void PreUpdate(const gz::sim::UpdateInfo &_info,
                        gz::sim::EntityComponentManager &_ecm) override;

  private: gz::sim::Entity linkEntity{gz::sim::kNullEntity};
  private: std::string linkName{"ball_link"};
  private: double mass{0.0027};
  private: double dragCoeff{0.1117};
  private: double magnusCoeff{0.0016};
  private: gz::math::Vector3d initLinear{6.0, 0.0, 0.9};
  private: gz::math::Vector3d initAngular{0.0, 400.0, 0.0};
  private: bool launched{false};
  private: bool cmdsCleared{false};
};
}  // namespace ttsim
#endif
