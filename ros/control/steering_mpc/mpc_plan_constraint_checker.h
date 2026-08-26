#pragma once

// NOTE: RoverState / TargetState are defined in the steering_mpc state header
// of the production tree. Adjust this include to wherever that type lives.
#include "sdg/sdc/ros/control/steering_mpc/rover_state.h"

#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

namespace yandex::sdc::control::steering_mpc {

    // Validates an MPC steering plan against the vehicle's kinematic comfort /
    // safety envelope before it is published. Two independent checks:
    //
    //   a) Normal (centripetal) acceleration of the vehicle (ВАТС) evaluated on
    //      the kinematic model from the resulting states:
    //          a_n = v^2 * tan(fwsa) / L,   v = hypot(velocity_x, velocity_y)
    //      taken from each RoverState in `results`. Must not exceed
    //      max_normal_acceleration in magnitude at any planned step.
    //
    //   b) Normal jerk, the time derivative of the normal acceleration from a):
    //          jerk_n[i] = (a_n(results[i + 1]) - a_n(results[i])) / time_step
    //      Must not exceed max_normal_jerk in magnitude on any step.
    //
    // IsPlanAdmissible() returns true when BOTH constraints hold everywhere, so
    // the caller can skip the trajectory when it returns false. `controls` is
    // accepted alongside `results` for a uniform plan interface; the checks
    // above are derived from the resulting states.
    class MpcPlanConstraintChecker {
    public:
        struct Limits {
            double max_normal_acceleration = 0.0; // m/s^2, > 0
            double max_normal_jerk = 0.0;         // m/s^3, > 0
            double wheelbase = 0.0;               // L, meters, > 0
            double time_step = 0.0;               // dt between plan points, s, > 0
        };

        struct Violation {
            enum class Kind {
                NormalAcceleration,
                NormalJerk,
            };

            Kind kind = Kind::NormalAcceleration;
            std::size_t index = 0; // plan step (state index / control index)
            double value = 0.0;    // signed offending value
            double limit = 0.0;    // limit it exceeded (positive)
        };

        explicit MpcPlanConstraintChecker(Limits limits) noexcept
            : limits_(limits)
        {
        }

        // Kinematic normal (centripetal) acceleration of a single planned state,
        // derived from its steering angle (fwsa): a_n = v^2 * tan(fwsa) / L.
        [[nodiscard]] double NormalAcceleration(const RoverState& state) const noexcept {
            const double speed = std::hypot(state.velocity_x, state.velocity_y);
            return speed * speed * std::tan(state.fwsa) / limits_.wheelbase;
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
            [[maybe_unused]] const std::vector<double>& controls) const noexcept {
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

            // b) Normal jerk: time derivative of the normal acceleration above.
            for (std::size_t i = 0; i + 1 < results.size(); ++i) {
                const double jerk =
                    (NormalAcceleration(results[i + 1]) - NormalAcceleration(results[i])) /
                    limits_.time_step;
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
        const Limits limits_;
    };

} // namespace yandex::sdc::control::steering_mpc
