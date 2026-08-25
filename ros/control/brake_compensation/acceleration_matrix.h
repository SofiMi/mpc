#pragma once

#include <cstddef>
#include <deque>
#include <optional>
#include <vector>

#include "ros/control/common/struct_interpolant.h"
#include "core/math/butterworth_filter.h"

namespace yandex::sdc::control {

    // Compensation table produced by BrakeCompensationBuilder.
    //
    // The table is a 2D grid keyed by (speed, acceleration). `value_points` stores
    // the multiplicative compensation coefficients in row-major order with the
    // acceleration index as the row and the speed index as the column:
    //
    //     value_points[acc_index * speed_points.size() + speed_index]
    //
    // The public shape of this struct (fields + BuildInterpolant) is part of the
    // external API and must stay stable.
    struct BrakeCompensationParams {
        std::vector<double> speed_points;
        std::vector<double> acceleration_points;
        std::vector<double> value_points;

        bool enable = false;

        StructInterpolant2d<ScalarAsArray<double>> BuildInterpolant() const;
    };

    // Online builder of the braking-compensation table.
    //
    // Algorithm (per the agreed spec):
    //   * Samples arrive one at a time via Set(); time advances by a fixed
    //     kSamplePeriod between calls (no wall-clock timestamps are used).
    //   * A braking event starts on the first sample with acceleration
    //     <= release_threshold_ (release_threshold_ is stored negative) and ends
    //     once acceleration returns to zero or above.
    //   * Only the monotonically deepening part of the event, from its start to the
    //     first local minimum (the peak of braking), is kept. Anything after the
    //     peak is ignored until the event ends.
    //   * Target and localization streams are processed fully independently: each
    //     produces its own event/profile, and the two are matched afterwards by
    //     normalized phase in [0, 1] (not by absolute time), which absorbs the
    //     known localization lag.
    //   * Because the two streams are filtered independently, one of them can
    //     silently drop an event (too short, never reached the release threshold,
    //     ...) while the other keeps it, which desynchronizes the two profile
    //     queues. Profiles are therefore matched FIFO only while their absolute
    //     start times stay within max_event_time_offset_ of each other; an orphan
    //     profile whose partner never showed up is discarded instead of being
    //     paired with an unrelated maneuver.
    //   * A profile pair is also discarded when the two events' durations are too
    //     disproportionate (ratio above max_duration_ratio_): the shorter profile
    //     is then a handful of raw samples stretched onto the fixed phase grid, so
    //     its per-phase values are dominated by interpolation/noise rather than the
    //     real maneuver and are not trustworthy enough to fold into the table.
    //   * Each profile is resampled onto a fixed phase grid; per-segment we take
    //     the absolute integral of acceleration. The per-segment coefficient is
    //     target_delta_I / localization_delta_I.
    //   * Coefficients are folded into the table with an exponential moving average
    //     (update_rate_), so recent maneuvers matter more than old ones.
    class BrakeCompensationBuilder {
    public:
        explicit BrakeCompensationBuilder(
            double release_threshold,
            double update_rate = 0.2,
            std::size_t smoothing_window = 5,
            double peak_release_margin = 0.2,
            std::size_t median_window = 5,
            double min_braking_duration = 0.5,
            double max_event_time_offset = 1.0,
            double max_duration_ratio = 3.0);

        // Feed one synchronized measurement. `localization_vel` is the current
        // speed, used both as the profile speed and as the table's speed key.
        void Set(
            double localization_acc,
            double target_acc,
            double localization_vel);

        // Drop all in-flight state and reset the table to its initial (all-ones)
        // grid.
        void Clear();

        // Current accumulated table.
        BrakeCompensationParams GetParams() const;

    public:
        // Resampled braking profile on the fixed phase grid.
        struct BrakeProfile {
            double begin_time = 0.0;            // absolute start time of the kept
                                                 // part, used to match this profile
                                                 // against its counterpart stream
            std::vector<double> phase;          // size kPhaseCount
            std::vector<double> speed;          // size kPhaseCount
            std::vector<double> acceleration;   // size kPhaseCount
            std::vector<double> delta_integral; // size kPhaseCount - 1
            double duration = 0.0;              // real-time length of the kept part
            std::vector<std::vector<double>> acceleration_segments;
            std::vector<std::vector<double>> time_segments;
        };

        struct DebugInfo {
            BrakeProfile target;
            BrakeProfile current;

            std::vector<double> vel;
            std::vector<double> acc;
            std::vector<double> coef_old;
            std::vector<double> coef_new;
            std::vector<double> coef_res;
        };

        std::optional<BrakeCompensationBuilder::DebugInfo> GetDebugInfo();

    private:
        struct Sample {
            double time = 0.0;
            double acc = 0.0;
            double velocity = 0.0;
        };

        // Monotonic braking segment (start .. peak inclusive).
        struct BrakeEvent {
            std::vector<Sample> samples;
        };

        // Streaming state of one (target or localization) braking event.
        struct ActiveEvent {
            std::vector<Sample> samples;
            bool active = false;
            bool peak_reached = false;
            std::size_t peak_index = 0;
        };

        static constexpr double kSamplePeriod = 0.02;
        static constexpr std::size_t kPhaseCount = 11; // phases 0.0 .. 1.0 step 0.1
        static constexpr double kEpsilon = 1e-9;

        void InitParams();

        // Feed a sample into a streaming event; on completion append its profile.
        void PushSample(
            ActiveEvent& event,
            const Sample& sample,
            std::deque<BrakeProfile>& completed_profiles);

        // Analyse a finished (buffered) event and, if usable, append its profile.
        void CompleteEvent(
            ActiveEvent& event,
            std::deque<BrakeProfile>& completed_profiles);

        // Zero-phase (centered) moving average of the samples' acceleration.
        std::vector<double> SmoothAcceleration(
            const std::vector<Sample>& samples) const;

        // First significant local minimum of `acc` using hysteresis (see class doc)
        // -- the index up to which the deepening part is kept.
        std::size_t FindPeakIndex(const std::vector<double>& acc) const;

        // Sliding-window median of a value series (spike-robust prefilter).
        std::vector<double> MedianFilter(const std::vector<double>& values) const;

        // Pair completed target/localization profiles (FIFO, guarded by start-time
        // proximity and duration ratio -- see class doc) and fold into the table.
        void TryUpdateParams();

        BrakeProfile BuildProfile(const BrakeEvent& event) const;

        double Interpolate(
            const std::vector<Sample>& samples,
            double time,
            bool velocity) const;

        double IntegrateAbsAcceleration(
            const std::vector<Sample>& samples,
            double begin_time,
            double end_time) const;

        void UpdateParams(
            const BrakeProfile& target_profile,
            const BrakeProfile& localization_profile);

        std::size_t FindNearestSpeedIndex(double speed) const;
        std::size_t FindNearestAccelerationIndex(double acceleration) const;

        double release_threshold_;
        double update_rate_;
        std::size_t smoothing_half_;      // effective window = 2 * half + 1
        double peak_release_margin_;
        std::size_t median_half_;         // effective window = 2 * half + 1
        double current_time_ = 0.0;

        ActiveEvent target_event_;
        ActiveEvent localization_event_;
        std::deque<BrakeProfile> target_profiles_;
        std::deque<BrakeProfile> localization_profiles_;

        BrakeCompensationParams params_;
        std::vector<double> update_counts_;
        std::optional<DebugInfo> debug_info_;
        double min_braking_duration_;
        double max_event_time_offset_;    // max allowed gap between matched
                                           // profiles' start times (see class doc)
        double max_duration_ratio_;       // max allowed longer/shorter duration
                                           // ratio between matched profiles
    };

} // namespace yandex::sdc::control
