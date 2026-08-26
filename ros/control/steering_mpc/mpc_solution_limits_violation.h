#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <vector>

namespace yandex::sdc::control::steering_mpc {

    // How much a single value exceeds a symmetric limit (0.0 if within it).
    inline double LimitViolation(double value, double limit) {
        return std::max(0.0, value - limit);
    }

    // Largest |visitor(item)| over the container. `ignore_first` skips the
    // leading element (used for transitions whose first entry is the seam with
    // the previous solution and must not count). Generic over both the element
    // type and the visitor, so one definition serves states and transitions.
    template <typename Item, typename Visitor>
        requires std::invocable<Visitor, const Item&>
    double MaxAbsoluteValue(
        const std::vector<Item>& items,
        const Visitor& visitor,
        bool ignore_first = false) {
        double max_value = 0.0;
        const std::size_t begin = ignore_first && !items.empty() ? 1 : 0;
        for (std::size_t i = begin; i < items.size(); ++i) {
            max_value = std::max(max_value, std::abs(visitor(items[i])));
        }
        return max_value;
    }

    // Max amount by which the visited quantity breaks `limit` across the
    // container, or 0.0 if it never does. Replaces the per-container overloads:
    // works for a vector of any element type (states, transitions, ...) and any
    // visitor, with the optional `ignore_first` seam skip.
    template <typename Item, typename Visitor>
        requires std::invocable<Visitor, const Item&>
    double MaxLimitViolation(
        const std::vector<Item>& items,
        const Visitor& visitor,
        double limit,
        bool ignore_first = false) {
        return LimitViolation(MaxAbsoluteValue(items, visitor, ignore_first), limit);
    }

    // Same, but the quantity is limited at two endpoints of each element (a
    // transition's start and final): the reported violation is the worse of the
    // two. The two-visitor arity keeps this unambiguous with the single-visitor
    // overload above.
    template <typename Item, typename StartVisitor, typename FinalVisitor>
        requires std::invocable<StartVisitor, const Item&> &&
                 std::invocable<FinalVisitor, const Item&>
    double MaxLimitViolation(
        const std::vector<Item>& items,
        const StartVisitor& start_visitor,
        const FinalVisitor& final_visitor,
        double limit,
        bool ignore_first = false) {
        return std::max(
            MaxLimitViolation(items, start_visitor, limit, ignore_first),
            MaxLimitViolation(items, final_visitor, limit, ignore_first));
    }

} // namespace yandex::sdc::control::steering_mpc
