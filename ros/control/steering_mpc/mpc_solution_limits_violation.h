#pragma once

#include <algorithm>
#include <cmath>
#include <span>

namespace yandex::sdc::control::steering_mpc {

    // Largest amount by which the limited quantity exceeds `limit` anywhere
    // along the solution, or 0.0 when the limit is respected at every state.
    //
    // `projection` maps a solution state to the scalar being limited (fwsa, its
    // velocity, normal acceleration, ...); the limit is treated as symmetric, so
    // the magnitude is what is compared. A non-positive result means "no
    // violation" -- callers store 0.0 in that case, which is why the overshoot
    // is clamped at 0.
    template <typename State, typename Projection>
    [[nodiscard]] double MaxLimitViolation(
        std::span<const State> solution_states,
        Projection projection,
        double limit) noexcept {
        double worst = 0.0;
        for (const State& state : solution_states) {
            const double overshoot = std::abs(projection(state)) - limit;
            worst = std::max(worst, overshoot);
        }
        return worst;
    }

} // namespace yandex::sdc::control::steering_mpc
