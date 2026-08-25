#include "acceleration_matrix.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace yandex::sdc::control {
namespace {

// Builds one synthetic acceleration stream: a linear ramp from the entry
// threshold (-0.2) down to `peak_acc` over `ramp_len` samples (the
// monotonically deepening part), followed by a linear release back up past
// zero over `release_len` samples, which ends the braking event.
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

std::vector<double> ScaleSignal(const std::vector<double>& source, double scale) {
    std::vector<double> out(source.size());
    for (std::size_t i = 0; i < source.size(); ++i) {
        out[i] = scale * source[i];
    }
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

// The current design: no lag estimation at all. The comparison window starts
// where target's own smoothed signal crosses sync_threshold_, and
// localization is read at those exact same absolute times. Matching happens
// synchronously as soon as target's event completes.
TEST(BrakeCompensationBuilderTest, MatchedEventRecoversCoefficient) {
    BrakeCompensationBuilder builder(
        /*release_threshold=*/-0.5, /*update_rate=*/1.0);

    constexpr double kScale = 1.3;
    const auto base = BuildBrakingSignal(/*peak_acc=*/-1.5, 60, 15);
    const auto target = ScaleSignal(base, kScale);

    FeedStreams(builder, base, target, /*speed=*/5.0);

    const auto debug = builder.GetDebugInfo();
    ASSERT_TRUE(debug.has_value());
    ASSERT_FALSE(debug->coef_new.empty());
    for (const double coefficient : debug->coef_new) {
        EXPECT_NEAR(coefficient, kScale, 0.05);
    }
}

// The comparison window must start where the smoothed trend first reaches
// sync_threshold_ (-0.3 by default), not at the raw -0.2 activation time
// (t=0 here, since the signal's first sample is already <= -0.2).
TEST(BrakeCompensationBuilderTest, WindowStartsAtSyncThresholdNotEntryThreshold) {
    BrakeCompensationBuilder builder(/*release_threshold=*/-0.5, /*update_rate=*/1.0);
    const auto base = BuildBrakingSignal(/*peak_acc=*/-1.5, 60, 15);

    FeedStreams(builder, base, base, /*speed=*/5.0);

    const auto debug = builder.GetDebugInfo();
    ASSERT_TRUE(debug.has_value());
    EXPECT_GT(debug->target.begin_time, 0.05);
    EXPECT_LT(debug->target.begin_time, 0.3);
}

// Regression test for a real observed failure: target's kept samples are
// smoothed (median + moving average) in CompleteEvent before its profile is
// built, but the localization slice went straight into BuildProfileFromSamples
// raw. A single-sample noise spike in localization (measurement noise, not a
// real maneuver feature) then skewed that phase point's contribution to its
// table cell far more than a real trend would.
TEST(BrakeCompensationBuilderTest, NoisyLocalizationIsSmoothedBeforeMatching) {
    BrakeCompensationBuilder builder(
        /*release_threshold=*/-0.5, /*update_rate=*/1.0);

    constexpr double kScale = 1.3;
    const auto base = BuildBrakingSignal(/*peak_acc=*/-1.5, 60, 15);
    const auto target = ScaleSignal(base, kScale);

    // Localization measurement with isolated single-sample noise spikes
    // (toward zero) every few samples -- exactly what the median prefilter
    // in SmoothAcceleration is meant to discard, if it's actually applied.
    std::vector<double> localization = base;
    for (std::size_t i = 3; i < localization.size(); i += 5) {
        localization[i] *= 0.2;
    }

    FeedStreams(builder, localization, target, /*speed=*/5.0);

    const auto debug = builder.GetDebugInfo();
    ASSERT_TRUE(debug.has_value());
    ASSERT_FALSE(debug->coef_new.empty());
    for (const double coefficient : debug->coef_new) {
        EXPECT_NEAR(coefficient, kScale, 0.15);
    }
}

// Regression test for a real observed failure: localization drops out for
// the whole event, with only a short, deep-looking real burst well after it
// ends. Interpolate would otherwise clamp the whole comparison window to
// that one distant sample, which is deep enough to pass a plain magnitude
// check despite belonging to an unrelated moment. The window must instead be
// rejected for lacking continuous real coverage.
TEST(BrakeCompensationBuilderTest, GapDuringEventIsRejectedNotClampedIntoAMatch) {
    BrakeCompensationBuilder builder(/*release_threshold=*/-0.5, /*update_rate=*/1.0);

    const auto target = BuildBrakingSignal(/*peak_acc=*/-1.5, 60, 15);

    // Matching happens synchronously the moment target's event completes, so
    // localization can only ever contain data already fed by then -- a
    // "real burst far in the future" isn't reachable. The gap has to be
    // inside the already-fed range instead: real data early (before the
    // comparison window, which starts a few samples in, at the
    // sync_threshold_ crossing), then a long dropout spanning the whole
    // sync-to-peak window, then real (deep) data again right up to target's
    // own release-crossing completion. The deep run needs to be long enough
    // that the median prefilter doesn't treat it as a discardable outlier
    // (it's surrounded, in filter-index terms, by the early zeros) -- a
    // short deep run gets smoothed away and the plain magnitude check alone
    // would then reject the match anyway, without the coverage check ever
    // being exercised.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> localization(target.size(), nan);
    for (std::size_t i = 0; i < 10; ++i) {
        localization[i] = 0.0;
    }
    for (std::size_t i = 40; i < 69; ++i) {
        localization[i] = -0.9;
    }

    FeedStreams(builder, localization, target, /*speed=*/5.0);

    EXPECT_FALSE(builder.GetDebugInfo().has_value());
    for (const double value : builder.GetParams().value_points) {
        EXPECT_DOUBLE_EQ(value, 1.0);
    }
}

// If localization shows no signal at all during the event (dropout, or a
// genuinely dead stream), the whole window reads back as flat/shallow --
// not real braking -- so the update is dropped.
TEST(BrakeCompensationBuilderTest, EventWithNoLocalizationSignalIsDropped) {
    BrakeCompensationBuilder builder(/*release_threshold=*/-0.5, /*update_rate=*/1.0);

    const auto target = BuildBrakingSignal(/*peak_acc=*/-2.0, 60, 15);
    FeedStreams(builder, NeutralSignal(target.size()), target, /*speed=*/5.0);

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

    const auto base = BuildBrakingSignal(/*peak_acc=*/-2.0, 60, 15);
    const auto target = ScaleSignal(base, 1.3);

    FeedStreams(builder, base, target, /*speed=*/5.0);

    const auto debug = builder.GetDebugInfo();
    ASSERT_TRUE(debug.has_value());

    for (const BrakeCompensationBuilder::BrakeProfile* profile :
         {&debug->target, &debug->current}) {
        ASSERT_EQ(profile->acceleration_segments.size(), 10u);
        ASSERT_EQ(profile->time_segments.size(), 10u);

        for (std::size_t i = 0; i < profile->acceleration_segments.size(); ++i) {
            // The old bug doubled every entry in acceleration_segments only;
            // same length here is exactly what would have caught it.
            EXPECT_EQ(profile->acceleration_segments[i].size(), profile->time_segments[i].size())
                << "interval " << i;

            const double interval_begin =
                profile->begin_time + profile->phase[i] * profile->duration;
            const double interval_end =
                profile->begin_time + profile->phase[i + 1] * profile->duration;
            for (const double time : profile->time_segments[i]) {
                EXPECT_GE(time, interval_begin - 1e-9) << "interval " << i;
                EXPECT_LE(time, interval_end + 1e-9) << "interval " << i;
            }
        }
    }
}

TEST(BrakeCompensationBuilderTest, CoefficientAboveMaxIsClamped) {
    BrakeCompensationBuilder builder(
        /*release_threshold=*/-0.5, /*update_rate=*/1.0);

    const auto base = BuildBrakingSignal(/*peak_acc=*/-2.0, 60, 15);
    const auto target = ScaleSignal(base, 3.0); // far above the [1, 1.5] range

    FeedStreams(builder, base, target, /*speed=*/5.0);

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
        // A neutral gap so the next repeat's -0.2 crossing starts a fresh
        // event rather than continuing the tail of the release ramp.
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

// Regression test for: Clear() is documented to drop all in-flight state and
// reset the table, but used to only clear the event/queue state.
TEST(BrakeCompensationBuilderTest, ClearResetsTableToInitialAllOnesGrid) {
    BrakeCompensationBuilder builder(
        /*release_threshold=*/-0.5, /*update_rate=*/1.0);

    const auto base = BuildBrakingSignal(/*peak_acc=*/-2.0, 60, 15);
    FeedStreams(builder, base, ScaleSignal(base, 1.3), /*speed=*/5.0);

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

    // A fresh event afterwards must be picked up normally, proving no
    // in-flight event/history state leaked past Clear().
    const auto base2 = BuildBrakingSignal(/*peak_acc=*/-1.5, 60, 15);
    FeedStreams(builder, base2, ScaleSignal(base2, 1.2), /*speed=*/5.0);

    const auto debug = builder.GetDebugInfo();
    ASSERT_TRUE(debug.has_value());
    ASSERT_FALSE(debug->coef_new.empty());
    for (const double coefficient : debug->coef_new) {
        EXPECT_NEAR(coefficient, 1.2, 0.05);
    }
}

} // namespace
} // namespace yandex::sdc::control
