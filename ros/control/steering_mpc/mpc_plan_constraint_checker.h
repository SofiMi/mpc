#pragma once

// NOTE: RoverState / TargetState are defined in the steering_mpc state header
// of the production tree. Adjust this include to wherever that type lives.
#include "sdg/sdc/ros/control/steering_mpc/rover_state.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

namespace yandex::sdc::control::steering_mpc {

    // Validates an MPC steering plan against the vehicle's kinematic envelope
    // before it is published. Both checks invert the same speed-dependent
    // constraints the planner builds from max_kinematic_normal_acceleration /
    // max_kinematic_normal_jerk, so a plan that satisfies them here is exactly
    // one the constraint builder would have admitted:
    //
    //   a) Normal (centripetal) acceleration, from fwsa in the resulting states
    //      (max_abs_fwsa = atan(a_n * L / v^2) * slip inverted):
    //          a_n = v^2 * tan(fwsa / slip) / L
    //      with v = hypot(velocity_x, velocity_y), L = GetWheelBase(),
    //      slip = SlipCorrection(v). Must not exceed max_normal_acceleration.
    //
    //   b) Normal jerk, from the steering-rate controls (fwsa_rate)
    //      (max_fwsa_rate = jerk * (L / v^2) * slip inverted):
    //          jerk_n = v^2 * fwsa_rate / (L * slip)
    //      with the speed taken from the state the control applies at. Must not
    //      exceed max_normal_jerk.
    //
    // At v == 0 both values are 0 (the v^2 factor), so a standing plan is always
    // admissible and SlipCorrection is not queried.
    //
    // IsPlanAdmissible() returns true when both constraints hold everywhere, so
    // the caller can skip the trajectory when it returns false.
    //
    // Templated on the steering model to stay header-only and unit-testable; the
    // model only needs `double GetWheelBase() const` and
    // `double SlipCorrection(double speed) const`.
    template <typename SteeringModel>
    class MpcPlanConstraintChecker {
    public:
        struct Limits {
            double max_normal_acceleration = 0.0; // m/s^2, > 0
            double max_normal_jerk = 0.0;         // m/s^3, > 0
        };

        struct Violation {
            enum class Kind {
                NormalAcceleration,
                NormalJerk,
            };

            Kind kind = Kind::NormalAcceleration;
            std::size_t index = 0; // offending step (state index / control index)
            double value = 0.0;    // signed offending value
            double limit = 0.0;    // limit it exceeded (positive)
        };

        MpcPlanConstraintChecker(const SteeringModel& steering_model, Limits limits) noexcept
            : steering_model_(steering_model)
            , limits_(limits)
        {
        }

        // Kinematic normal (centripetal) acceleration of a single planned state,
        // from its steering angle: a_n = v^2 * tan(fwsa / slip) / L.
        [[nodiscard]] double NormalAcceleration(const RoverState& state) const noexcept {
            const double speed = std::hypot(state.velocity_x, state.velocity_y);
            if (speed <= 0.0) {
                return 0.0;
            }
            const double slip = steering_model_.SlipCorrection(speed);
            return speed * speed * std::tan(state.fwsa / slip) / steering_model_.GetWheelBase();
        }

        // Kinematic normal jerk implied by a steering-rate control at the given
        // state: jerk_n = v^2 * fwsa_rate / (L * slip).
        [[nodiscard]] double NormalJerk(const RoverState& state, double fwsa_rate) const noexcept {
            const double speed = std::hypot(state.velocity_x, state.velocity_y);
            if (speed <= 0.0) {
                return 0.0;
            }
            const double slip = steering_model_.SlipCorrection(speed);
            return speed * speed * fwsa_rate / (steering_model_.GetWheelBase() * slip);
        }

        // true  -> plan is admissible, publish it.
        // false -> some constraint is violated, skip the trajectory.
        [[nodiscard]] bool IsPlanAdmissible(
            const std::vector<RoverState>& results,
            const std::vector<double>& controls) const noexcept {
            return !FindViolation(results, controls).has_value();
        }

        // First offending step, if any (normal acceleration is checked before
        // normal jerk). Handy for diagnostics / logging why a plan was skipped.
        [[nodiscard]] std::optional<Violation> FindViolation(
            const std::vector<RoverState>& results,
            const std::vector<double>& controls) const noexcept {
            // a) Normal acceleration by fwsa from the resulting states.
            for (std::size_t i = 0; i < results.size(); ++i) {
                const double a_n = NormalAcceleration(results[i]);
                if (std::abs(a_n) > limits_.max_normal_acceleration) {
                    return Violation{
                        .kind = Violation::Kind::NormalAcceleration,
                        .index = i,
                        .value = a_n,
                        .limit = limits_.max_normal_acceleration,
                    };
                }
            }

            // b) Normal jerk from the steering-rate controls; each control
            // applies at the state with the same index.
            const std::size_t jerk_steps = std::min(controls.size(), results.size());
            for (std::size_t i = 0; i < jerk_steps; ++i) {
                const double jerk = NormalJerk(results[i], controls[i]);
                if (std::abs(jerk) > limits_.max_normal_jerk) {
                    return Violation{
                        .kind = Violation::Kind::NormalJerk,
                        .index = i,
                        .value = jerk,
                        .limit = limits_.max_normal_jerk,
                    };
                }
            }

            return std::nullopt;
        }

    private:
        const SteeringModel& steering_model_;
        const Limits limits_;
    };

} // namespace yandex::sdc::control::steering_mpc
