#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <ranges>

namespace yandex::sdc::control::steering_mpc {

    // How much a single value exceeds a symmetric limit (0.0 if within it).
    inline double LimitViolation(double value, double limit) {
        return std::max(0.0, value - limit);
    }

    // Largest |visitor(item)| over the sequence. `ignore_first` skips the
    // leading element (a transition/seam that must not count). Generic over the
    // container (std::vector, std::span, ...), the element type, and the
    // visitor, so one definition serves every limited quantity.
    template <std::ranges::input_range Range, typename Visitor>
        requires std::invocable<Visitor, const std::ranges::range_value_t<Range>&>
    double MaxAbsoluteValue(
        const Range& items,
        const Visitor& visitor,
        bool ignore_first = false) {
        double max_value = 0.0;
        auto it = std::ranges::begin(items);
        const auto end = std::ranges::end(items);
        if (ignore_first && it != end) {
            ++it;
        }
        for (; it != end; ++it) {
            max_value = std::max(max_value, std::abs(visitor(*it)));
        }
        return max_value;
    }

    // Max amount by which the visited quantity breaks `limit` across the
    // sequence, or 0.0 if it never does.
    template <std::ranges::input_range Range, typename Visitor>
        requires std::invocable<Visitor, const std::ranges::range_value_t<Range>&>
    double MaxLimitViolation(
        const Range& items,
        const Visitor& visitor,
        double limit,
        bool ignore_first = false) {
        return LimitViolation(MaxAbsoluteValue(items, visitor, ignore_first), limit);
    }

    // Same, but the quantity is limited at two endpoints of each element: the
    // reported violation is the worse of the two. The two-visitor arity keeps
    // this unambiguous with the single-visitor overload above.
    template <std::ranges::input_range Range, typename StartVisitor, typename FinalVisitor>
        requires std::invocable<StartVisitor, const std::ranges::range_value_t<Range>&> &&
                 std::invocable<FinalVisitor, const std::ranges::range_value_t<Range>&>
    double MaxLimitViolation(
        const Range& items,
        const StartVisitor& start_visitor,
        const FinalVisitor& final_visitor,
        double limit,
        bool ignore_first = false) {
        return std::max(
            MaxLimitViolation(items, start_visitor, limit, ignore_first),
            MaxLimitViolation(items, final_visitor, limit, ignore_first));
    }

} // namespace yandex::sdc::control::steering_mpc
