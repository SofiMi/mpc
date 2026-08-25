#include "acceleration_matrix.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace yandex::sdc::control {
namespace {

constexpr double kSamplePeriod = 0.02;

// Builds one synthetic acceleration stream: a linear ramp from the entry
// threshold (-0.2) down to `peak_acc` over `ramp_len` samples (the
// monotonically deepening part that ends up kept in the profile), followed
// by a linear release back up past zero over `release_len` samples, which
// ends the braking event.
std::vector<double> BuildBrakingSignal(
    double peak_acc, std::size_t ramp_len, std::size_t release_len) {
    std::vector<double> signal;
    signal.reserve(ramp_len + release_len);
    for (std::size_t i = 0; i < ramp_len; ++i) {
        const double t = ramp_len > 1
            ? static_cast<double>(i) / static_cast<double>(ramp_len - 1)
            : 1.0;
        signal.push_back(-0.2 + t * (peak_acc + 0.2));
    }
    for (std::size_t i = 0; i < release_len; ++i) {
        const double t =
            static_cast<double>(i + 1) / static_cast<double>(release_len);
        signal.push_back(peak_acc + t * (0.5 - peak_acc));
    }
    return signal;
}

std::vector<double> NeutralSignal(std::size_t length) {
    return std::vector<double>(length, 0.0);
}

// Feeds two independently-sized streams through the builder, one sample per
// Set() call, at a constant speed; the shorter stream is padded with
// neutral (0.0) samples.
void FeedStreams(
    BrakeCompensationBuilder& builder,
    const std::vector<double>& localization_acc,
    const std::vector<double>& target_acc,
    double speed) {
    const std::size_t n = std::max(localization_acc.size(), target_acc.size());
    for (std::size_t i = 0; i < n; ++i) {
        const double loc = i < localization_acc.size() ? localization_acc[i] : 0.0;
        const double tgt = i < target_acc.size() ? target_acc[i] : 0.0;
        builder.Set(loc, tgt, speed);
    }
}

TEST(BrakeCompensationBuilderTest, InitialTableIsAllOnes) {
    BrakeCompensationBuilder builder(/*release_threshold=*/-0.5);
    const BrakeCompensationParams params = builder.GetParams();

    EXPECT_TRUE(params.enable);
    ASSERT_EQ(params.speed_points.size(), 11u);
    ASSERT_EQ(params.acceleration_points.size(), 9u);
    ASSERT_EQ(params.value_points.size(), 11u * 9u);
    for (const double value : params.value_points) {
        EXPECT_DOUBLE_EQ(value, 1.0);
    }
}

TEST(BrakeCompensationBuilderTest, NeutralDrivingNeverUpdatesTable) {
    BrakeCompensationBuilder builder(/*release_threshold=*/-0.5);
    for (int i = 0; i < 500; ++i) {
        builder.Set(
            /*localization_acc=*/0.0, /*target_acc=*/0.0, /*localization_vel=*/5.0);
    }
    for (const double value : builder.GetParams().value_points) {
        EXPECT_DOUBLE_EQ(value, 1.0);
    }
    EXPECT_FALSE(builder.GetDebugInfo().has_value());
}

TEST(BrakeCompensationBuilderTest, NonFiniteSamplesAreIgnored) {
    BrakeCompensationBuilder builder(/*release_threshold=*/-0.5);
    const double nan_value = std::numeric_limits<double>::quiet_NaN();
    for (int i = 0; i < 10; ++i) {
        builder.Set(nan_value, nan_value, nan_value);
    }
    for (const double value : builder.GetParams().value_points) {
        EXPECT_DOUBLE_EQ(value, 1.0);
    }
    EXPECT_FALSE(builder.GetDebugInfo().has_value());
}

TEST(BrakeCompensationBuilderTest, MatchedEventWithUnitUpdateRateSetsCoefficient) {
    // update_rate = 1.0 makes the EMA a direct assignment, so the resulting
    // table value is exactly the (clamped) coefficient of the single update.
    BrakeCompensationBuilder builder(
        /*release_threshold=*/-0.5, /*update_rate=*/1.0);

    const auto localization = BuildBrakingSignal(/*peak_acc=*/-2.0, 60, 15);
    std::vector<double> target(localization.size());
    for (std::size_t i = 0; i < localization.size(); ++i) {
        target[i] = 1.3 * localization[i];
    }

    FeedStreams(builder, localization, target, /*speed=*/5.0);

    const auto debug = builder.GetDebugInfo();
    ASSERT_TRUE(debug.has_value());
    ASSERT_FALSE(debug->coef_new.empty());
    for (const double coefficient : debug->coef_new) {
        EXPECT_NEAR(coefficient, 1.3, 0.05);
    }

    bool any_updated = false;
    for (const double value : builder.GetParams().value_points) {
        EXPECT_GE(value, 1.0);
        EXPECT_LE(value, 1.5);
        if (value > 1.0) {
            any_updated = true;
            EXPECT_NEAR(value, 1.3, 0.05);
        }
    }
    EXPECT_TRUE(any_updated);
}

TEST(BrakeCompensationBuilderTest, CoefficientAboveMaxIsClamped) {
    BrakeCompensationBuilder builder(
        /*release_threshold=*/-0.5, /*update_rate=*/1.0);

    const auto localization = BuildBrakingSignal(/*peak_acc=*/-2.0, 60, 15);
    std::vector<double> target(localization.size());
    for (std::size_t i = 0; i < localization.size(); ++i) {
        target[i] = 3.0 * localization[i]; // far above the [1, 1.5] range
    }

    FeedStreams(builder, localization, target, /*speed=*/5.0);

    bool any_updated = false;
    for (const double value : builder.GetParams().value_points) {
        ASSERT_GE(value, 1.0);
        ASSERT_LE(value, 1.5);
        if (value > 1.0) {
            any_updated = true;
            EXPECT_NEAR(value, 1.5, 1e-6);
        }
    }
    EXPECT_TRUE(any_updated);
}

TEST(BrakeCompensationBuilderTest, RepeatedEventsConvergeViaExponentialMovingAverage) {
    BrakeCompensationBuilder builder(
        /*release_threshold=*/-0.5, /*update_rate=*/0.2);

    const auto localization = BuildBrakingSignal(/*peak_acc=*/-2.0, 60, 15);
    std::vector<double> target(localization.size());
    for (std::size_t i = 0; i < localization.size(); ++i) {
        target[i] = 1.3 * localization[i];
    }

    double previous_max = 1.0;
    for (int repeat = 0; repeat < 40; ++repeat) {
        FeedStreams(builder, localization, target, /*speed=*/5.0);
        // A gap between maneuvers so each is its own event/profile.
        FeedStreams(builder, NeutralSignal(20), NeutralSignal(20), 5.0);

        double current_max = 1.0;
        for (const double value : builder.GetParams().value_points) {
            current_max = std::max(current_max, value);
        }
        EXPECT_GE(current_max, previous_max - 1e-9);
        previous_max = current_max;
    }

    EXPECT_NEAR(previous_max, 1.3, 0.05);
}

// Regression test for: independently-filtered streams can desync (one
// stream drops an event the other kept), which used to make TryUpdateParams
// pair unrelated maneuvers together forever after.
TEST(BrakeCompensationBuilderTest, OrphanProfileAcrossLargeGapIsNotPairedWithLaterEvent) {
    constexpr double kMaxEventTimeOffset = 1.0; // matches the default
    BrakeCompensationBuilder builder(
        /*release_threshold=*/-0.5,
        /*update_rate=*/1.0,
        /*smoothing_window=*/5,
        /*peak_release_margin=*/0.2,
        /*median_window=*/5,
        /*min_braking_duration=*/0.5,
        /*max_event_time_offset=*/kMaxEventTimeOffset,
        /*max_duration_ratio=*/3.0);

    // Maneuver A: only the target brakes; localization stays well above the
    // entry threshold the whole time, so it never produces a profile.
    const auto target_a = BuildBrakingSignal(/*peak_acc=*/-3.5, 40, 15);
    FeedStreams(builder, std::vector<double>(target_a.size(), -0.05), target_a, 5.0);

    // A neutral gap far larger than max_event_time_offset_.
    const std::size_t gap_samples =
        static_cast<std::size_t>((kMaxEventTimeOffset * 5.0) / kSamplePeriod);
    FeedStreams(builder, NeutralSignal(gap_samples), NeutralSignal(gap_samples), 5.0);

    // Maneuver B: both streams brake together with an identical signal, so
    // they are perfectly matched (coefficient must be exactly 1).
    const auto shared_b = BuildBrakingSignal(/*peak_acc=*/-1.5, 40, 15);

    std::vector<BrakeCompensationBuilder::DebugInfo> updates;
    for (std::size_t i = 0; i < shared_b.size(); ++i) {
        builder.Set(shared_b[i], shared_b[i], /*localization_vel=*/5.0);
        const auto debug = builder.GetDebugInfo();
        if (debug.has_value() &&
            (updates.empty() ||
             updates.back().target.begin_time != debug->target.begin_time)) {
            updates.push_back(*debug);
        }
    }

    // Exactly one real pairing happened (maneuver B with itself); the
    // target-only orphan from maneuver A was discarded rather than being
    // paired with B once B's localization profile finally showed up.
    ASSERT_EQ(updates.size(), 1u);
    EXPECT_LE(
        std::abs(updates[0].target.begin_time - updates[0].current.begin_time),
        kMaxEventTimeOffset);
    for (const double coefficient : updates[0].coef_new) {
        EXPECT_NEAR(coefficient, 1.0, 1e-6);
    }
}

// Regression test for: a matched pair whose durations are wildly different
// (the shorter one under-resolved on the fixed phase grid) used to still be
// folded into the table.
TEST(BrakeCompensationBuilderTest, GrosslyDisproportionateDurationPairIsDropped) {
    BrakeCompensationBuilder builder(
        /*release_threshold=*/-0.5,
        /*update_rate=*/1.0,
        /*smoothing_window=*/5,
        /*peak_release_margin=*/0.2,
        /*median_window=*/5,
        /*min_braking_duration=*/0.5,
        /*max_event_time_offset=*/1.0,
        /*max_duration_ratio=*/3.0);

    // Localization brakes only briefly (just above min_braking_duration);
    // target brakes for far longer. Both start at the same time, so the
    // start-time guard alone would let them pair.
    const auto localization = BuildBrakingSignal(/*peak_acc=*/-1.5, 35, 15);
    const auto target = BuildBrakingSignal(/*peak_acc=*/-1.5, 170, 15);

    FeedStreams(builder, localization, target, /*speed=*/5.0);

    EXPECT_FALSE(builder.GetDebugInfo().has_value());
    for (const double value : builder.GetParams().value_points) {
        EXPECT_DOUBLE_EQ(value, 1.0);
    }
}

TEST(BrakeCompensationBuilderTest, ModeratelyDifferentDurationPairIsStillAccepted) {
    BrakeCompensationBuilder builder(
        /*release_threshold=*/-0.5,
        /*update_rate=*/1.0,
        /*smoothing_window=*/5,
        /*peak_release_margin=*/0.2,
        /*median_window=*/5,
        /*min_braking_duration=*/0.5,
        /*max_event_time_offset=*/1.0,
        /*max_duration_ratio=*/3.0);

    const auto localization = BuildBrakingSignal(/*peak_acc=*/-1.5, 35, 15);
    const auto target = BuildBrakingSignal(/*peak_acc=*/-1.5, 60, 15);

    FeedStreams(builder, localization, target, /*speed=*/5.0);

    const auto debug = builder.GetDebugInfo();
    ASSERT_TRUE(debug.has_value());
    const double ratio =
        std::max(debug->target.duration, debug->current.duration) /
        std::min(debug->target.duration, debug->current.duration);
    EXPECT_LE(ratio, 3.0);
}

// Regression test for: Clear() is documented to reset the table to its
// initial all-ones grid but used to only clear in-flight event/queue state.
TEST(BrakeCompensationBuilderTest, ClearResetsTableToInitialAllOnesGrid) {
    BrakeCompensationBuilder builder(
        /*release_threshold=*/-0.5, /*update_rate=*/1.0);

    const auto localization = BuildBrakingSignal(/*peak_acc=*/-2.0, 60, 15);
    std::vector<double> target(localization.size());
    for (std::size_t i = 0; i < localization.size(); ++i) {
        target[i] = 1.3 * localization[i];
    }
    FeedStreams(builder, localization, target, /*speed=*/5.0);

    bool any_updated = false;
    for (const double value : builder.GetParams().value_points) {
        any_updated = any_updated || value > 1.0;
    }
    ASSERT_TRUE(any_updated);

    builder.Clear();

    const BrakeCompensationParams params = builder.GetParams();
    EXPECT_TRUE(params.enable);
    for (const double value : params.value_points) {
        EXPECT_DOUBLE_EQ(value, 1.0);
    }
}

} // namespace
} // namespace yandex::sdc::control
