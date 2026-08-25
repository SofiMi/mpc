#include "acceleration_matrix.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace yandex::sdc::control {
namespace {

constexpr double kSamplePeriod = 0.02;
// Comfortably larger than the default max_lag_ (2.0s): enough neutral
// padding after an event that ProcessPendingTargetEvents is guaranteed to
// stop waiting and attempt (or reject) the match.
constexpr std::size_t kSettleSamples = 200; // 4.0s

// Builds one synthetic acceleration stream: a linear ramp from the entry
// threshold (-0.2) down to `peak_acc` over `ramp_len` samples (the
// monotonically deepening part that ends up kept in the target profile),
// followed by a linear release back up past zero over `release_len`
// samples, which ends the braking event.
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

constexpr double kLocalizationEntryThreshold = -0.3; // must match the default

// Same ramp+release shape as BuildBrakingSignal, but starting exactly at
// kLocalizationEntryThreshold instead of -0.2: a delayed copy's onset
// crossing then lands exactly on its first sample, giving exact control
// over the lag EstimateLag will find in tests (rather than the ramp's own
// -0.2-to-threshold travel time adding an unknown offset).
std::vector<double> BuildLocalizationSignal(
    double peak_acc, std::size_t ramp_len, std::size_t release_len) {
    std::vector<double> signal;
    signal.reserve(ramp_len + release_len);
    for (std::size_t i = 0; i < ramp_len; ++i) {
        const double t = ramp_len > 1
            ? static_cast<double>(i) / static_cast<double>(ramp_len - 1)
            : 1.0;
        signal.push_back(
            kLocalizationEntryThreshold + t * (peak_acc - kLocalizationEntryThreshold));
    }
    for (std::size_t i = 0; i < release_len; ++i) {
        const double t =
            static_cast<double>(i + 1) / static_cast<double>(release_len);
        signal.push_back(peak_acc + t * (0.5 - peak_acc));
    }
    return signal;
}

std::vector<double> ScaleSignal(const std::vector<double>& source, double scale) {
    std::vector<double> out(source.size());
    for (std::size_t i = 0; i < source.size(); ++i) {
        out[i] = scale * source[i];
    }
    return out;
}

// Simulates localization observing the same physical event `delay_samples`
// later than target: `delay_samples` neutral samples, then `source` as-is.
std::vector<double> DelaySignal(
    const std::vector<double>& source, std::size_t delay_samples) {
    std::vector<double> out(delay_samples, 0.0);
    out.insert(out.end(), source.begin(), source.end());
    return out;
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

// Enough neutral samples that any pending target event is past its
// max_lag_ wait and ProcessPendingTargetEvents will have matched (or
// rejected) it.
void FeedSettleGap(BrakeCompensationBuilder& builder, double speed) {
    FeedStreams(builder, NeutralSignal(kSettleSamples), NeutralSignal(kSettleSamples), speed);
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

// The central new behavior: target and localization no longer need to be
// sampled at the same absolute time. Localization observes the same
// maneuver `delay` seconds later (and here, weaker); the builder must
// discover that delay by itself (via localization's own entry-threshold
// crossing) and still recover the true amplitude coefficient once it
// corrects for it.
TEST(BrakeCompensationBuilderTest, DelayedAndScaledEventRecoversLagAndCoefficient) {
    BrakeCompensationBuilder builder(
        /*release_threshold=*/-0.5, /*update_rate=*/1.0);

    constexpr std::size_t kDelaySamples = 25; // 0.5s
    constexpr double kTrueLag = kDelaySamples * kSamplePeriod;
    constexpr double kScale = 1.3;

    const auto localization_base = BuildLocalizationSignal(/*peak_acc=*/-1.5, 60, 15);
    const auto target = ScaleSignal(localization_base, kScale);       // on time
    const auto localization = DelaySignal(localization_base, kDelaySamples); // late

    FeedStreams(builder, localization, target, /*speed=*/5.0);
    FeedSettleGap(builder, /*speed=*/5.0);

    const auto debug = builder.GetDebugInfo();
    ASSERT_TRUE(debug.has_value());
    EXPECT_NEAR(debug->lag, kTrueLag, 1e-6);
    EXPECT_DOUBLE_EQ(debug->lag_confidence, 1.0); // a fresh crossing was found

    ASSERT_FALSE(debug->coef_new.empty());
    for (const double coefficient : debug->coef_new) {
        EXPECT_NEAR(coefficient, kScale, 0.05);
    }
}

// Two maneuvers with clearly different real delays (e.g. two different
// cars, or the same car's lag simply not being a fixed constant): each
// must be recovered on its own terms, not dragged toward the other.
TEST(BrakeCompensationBuilderTest, LagIsReEstimatedPerEventNotStuckAtFirstValue) {
    BrakeCompensationBuilder builder(
        /*release_threshold=*/-0.5, /*update_rate=*/1.0);

    const auto base1 = BuildLocalizationSignal(/*peak_acc=*/-1.5, 60, 15);
    constexpr std::size_t kDelay1 = 20; // 0.4s
    FeedStreams(
        builder, DelaySignal(base1, kDelay1), ScaleSignal(base1, 1.2), /*speed=*/5.0);
    FeedSettleGap(builder, 5.0);

    const auto debug1 = builder.GetDebugInfo();
    ASSERT_TRUE(debug1.has_value());
    EXPECT_NEAR(debug1->lag, kDelay1 * kSamplePeriod, 1e-6);

    const auto base2 = BuildLocalizationSignal(/*peak_acc=*/-2.5, 60, 15);
    constexpr std::size_t kDelay2 = 60; // 1.2s
    FeedStreams(
        builder, DelaySignal(base2, kDelay2), ScaleSignal(base2, 1.2), /*speed=*/5.0);
    FeedSettleGap(builder, 5.0);

    const auto debug2 = builder.GetDebugInfo();
    ASSERT_TRUE(debug2.has_value());
    EXPECT_NEAR(debug2->lag, kDelay2 * kSamplePeriod, 1e-6);
}

// Regression test for a real observed failure: the true lag exceeds
// max_lag_, so localization's real entry crossing is never found within the
// search window. The event must be dropped instead of being matched against
// whatever's in range (e.g. an unrelated, still-shallow stretch), which is
// what used to produce a low-confidence, essentially made-up coefficient.
TEST(BrakeCompensationBuilderTest, LocalizationCrossingBeyondMaxLagIsNotUsed) {
    BrakeCompensationBuilder builder(
        /*release_threshold=*/-0.5,
        /*update_rate=*/1.0,
        /*smoothing_window=*/5,
        /*peak_release_margin=*/0.2,
        /*median_window=*/5,
        /*min_braking_duration=*/0.5,
        /*min_lag=*/0.0,
        /*max_lag=*/1.0); // narrow on purpose

    const auto localization_base = BuildLocalizationSignal(/*peak_acc=*/-1.5, 60, 15);
    const auto target = ScaleSignal(localization_base, 1.3);
    constexpr std::size_t kDelaySamples = 80; // 1.6s -- beyond max_lag_ = 1.0
    const auto localization = DelaySignal(localization_base, kDelaySamples);

    FeedStreams(builder, localization, target, /*speed=*/5.0);
    FeedSettleGap(builder, 5.0);

    EXPECT_FALSE(builder.GetDebugInfo().has_value());
    for (const double value : builder.GetParams().value_points) {
        EXPECT_DOUBLE_EQ(value, 1.0);
    }
}

// If localization shows no correlated signal anywhere in the search range
// (dropout, or a genuinely dead stream), the event must not be matched by
// chance: no confident lag estimate exists, the fallback (current_lag_,
// still 0 on a fresh builder) is tried, and the resulting window shows no
// real braking either -- so the whole update is dropped.
TEST(BrakeCompensationBuilderTest, EventWithNoLocalizationSignalIsDropped) {
    BrakeCompensationBuilder builder(/*release_threshold=*/-0.5, /*update_rate=*/1.0);

    const auto target = BuildBrakingSignal(/*peak_acc=*/-2.0, 60, 15);
    FeedStreams(builder, NeutralSignal(target.size()), target, /*speed=*/5.0);
    FeedSettleGap(builder, 5.0);

    EXPECT_FALSE(builder.GetDebugInfo().has_value());
    for (const double value : builder.GetParams().value_points) {
        EXPECT_DOUBLE_EQ(value, 1.0);
    }
}

// Regression test for: BuildProfile used to populate acceleration_segments
// twice (once in its own loop, once again while building time_segments in a
// second "for the same purpose" loop) but time_segments only once, so the
// two ended up different lengths and not index-aligned. BuildProfileFromSamples
// now fills both from a single loop, one push per sample.
TEST(BrakeCompensationBuilderTest, SegmentsAreIndexAlignedWithMatchingSamples) {
    BrakeCompensationBuilder builder(
        /*release_threshold=*/-0.5, /*update_rate=*/1.0);

    constexpr std::size_t kRampLen = 60;
    const auto base = BuildBrakingSignal(/*peak_acc=*/-2.0, kRampLen, 15);
    const auto target = ScaleSignal(base, 1.3);

    FeedStreams(builder, base, target, /*speed=*/5.0);
    FeedSettleGap(builder, 5.0);

    const auto debug = builder.GetDebugInfo();
    ASSERT_TRUE(debug.has_value());

    for (const BrakeCompensationBuilder::BrakeProfile* profile :
         {&debug->target, &debug->current}) {
        ASSERT_EQ(profile->acceleration_segments.size(), 10u);
        ASSERT_EQ(profile->time_segments.size(), 10u);

        std::size_t total_samples = 0;
        for (std::size_t i = 0; i < profile->acceleration_segments.size(); ++i) {
            // The old bug doubled every entry in acceleration_segments only;
            // same length here is exactly what would have caught it.
            EXPECT_EQ(profile->acceleration_segments[i].size(), profile->time_segments[i].size())
                << "interval " << i;
            total_samples += profile->time_segments[i].size();

            const double interval_begin =
                profile->begin_time + profile->phase[i] * profile->duration;
            const double interval_end =
                profile->begin_time + profile->phase[i + 1] * profile->duration;
            for (const double time : profile->time_segments[i]) {
                EXPECT_GE(time, interval_begin - 1e-9) << "interval " << i;
                EXPECT_LE(time, interval_end + 1e-9) << "interval " << i;
            }
        }
        // The target profile's window holds its kept raw samples, i.e. close
        // to kRampLen (smoothing can shift the detected peak by a few
        // samples either way); the old bug doubled every entry, which would
        // land near kRampLen * 2 here instead -- nowhere close to this bound.
        if (profile == &debug->target) {
            EXPECT_LE(total_samples, kRampLen);
            EXPECT_GE(total_samples, kRampLen - 5);
        }
    }
}

TEST(BrakeCompensationBuilderTest, CoefficientAboveMaxIsClamped) {
    BrakeCompensationBuilder builder(
        /*release_threshold=*/-0.5, /*update_rate=*/1.0);

    const auto base = BuildBrakingSignal(/*peak_acc=*/-2.0, 60, 15);
    const auto target = ScaleSignal(base, 3.0); // far above the [1, 1.5] range

    FeedStreams(builder, base, target, /*speed=*/5.0);
    FeedSettleGap(builder, 5.0);

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

    const auto base = BuildBrakingSignal(/*peak_acc=*/-2.0, 60, 15);
    const auto target = ScaleSignal(base, 1.3);

    double previous_max = 1.0;
    for (int repeat = 0; repeat < 10; ++repeat) {
        FeedStreams(builder, base, target, /*speed=*/5.0);
        FeedSettleGap(builder, 5.0);

        double current_max = 1.0;
        for (const double value : builder.GetParams().value_points) {
            current_max = std::max(current_max, value);
        }
        EXPECT_GE(current_max, previous_max - 1e-9);
        previous_max = current_max;
    }

    EXPECT_NEAR(previous_max, 1.3, 0.05);
}

// Regression test for: Clear() is documented to drop all in-flight state
// (including whatever lag has been learned) and reset the table, but used
// to only clear the event/queue state.
TEST(BrakeCompensationBuilderTest, ClearResetsTableAndLearnedLag) {
    BrakeCompensationBuilder builder(
        /*release_threshold=*/-0.5, /*update_rate=*/1.0);

    const auto base1 = BuildBrakingSignal(/*peak_acc=*/-2.0, 60, 15);
    FeedStreams(builder, base1, ScaleSignal(base1, 1.3), /*speed=*/5.0);
    FeedSettleGap(builder, 5.0);

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

    // A fresh event with a different delay must be picked up on its own
    // terms, proving no localization history/lag state leaked past Clear().
    const auto base2 = BuildLocalizationSignal(/*peak_acc=*/-1.5, 60, 15);
    constexpr std::size_t kDelay = 30; // 0.6s
    FeedStreams(
        builder, DelaySignal(base2, kDelay), ScaleSignal(base2, 1.2), /*speed=*/5.0);
    FeedSettleGap(builder, 5.0);

    const auto debug = builder.GetDebugInfo();
    ASSERT_TRUE(debug.has_value());
    EXPECT_NEAR(debug->lag, kDelay * kSamplePeriod, 1e-6);
}

} // namespace
} // namespace yandex::sdc::control
