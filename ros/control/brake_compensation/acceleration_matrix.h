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
    //   * Only the target stream drives event detection: a braking event starts
    //     on the first sample with acceleration <= release_threshold_
    //     (release_threshold_ is stored negative) and ends once acceleration
    //     returns to zero or above. Only the monotonically deepening part, from
    //     the start to the first local minimum (the peak of braking), is kept.
    //   * The localization stream is never independently thresholded or peak-
    //     detected -- its raw samples are just kept in a rolling history buffer
    //     (localization_history_). Detecting braking independently on both
    //     streams and then matching by normalized phase used to be how this
    //     worked, but that conflates two unrelated things: the real, physical
    //     lag between the streams, and each stream's own onset-detection noise
    //     (a noisier signal crosses its own threshold later and already deeper
    //     than it should be) -- which silently distorted the phase-0 point.
    //   * Instead, once a target event's kept (monotonic) window
    //     [begin_time, peak_time] is known, the matching localization window is
    //     [begin_time + lag, peak_time + lag] for an explicitly estimated time
    //     lag: for each event, a candidate lag in [min_lag_, max_lag_] is chosen
    //     by normalized cross-correlation between the target event's full
    //     buffered shape (deepening + release, which has a real minimum to
    //     anchor on, unlike the plain monotonic ramp) and the localization
    //     history shifted by that candidate. A confident per-event estimate
    //     updates current_lag_ (an EMA, so a car's actual brake/estimation lag
    //     is learned over time instead of assumed); a low-confidence one falls
    //     back to the current current_lag_ without updating it. Either way, the
    //     resulting localization window is sanity-checked (it must show real
    //     braking, i.e. dip at or below release_threshold_) before the pair is
    //     used, so a bad match cannot corrupt the table.
    //   * Because the localization profile is always built over a window whose
    //     length equals the target profile's, there is no independent
    //     localization duration to mismatch, and no separate profile queue to
    //     desynchronize -- a class of bugs the old two-sided independent
    //     detection was prone to.
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
            double min_lag = 0.0,
            double max_lag = 2.0,
            double lag_update_rate = 0.2,
            double min_lag_correlation = 0.5);

        // Feed one synchronized measurement. `localization_vel` is the current
        // speed, used both as the profile speed and as the table's speed key.
        void Set(
            double localization_acc,
            double target_acc,
            double localization_vel);

        // Drop all in-flight state (including the learned lag) and reset the
        // table to its initial (all-ones) grid.
        void Clear();

        // Current accumulated table.
        BrakeCompensationParams GetParams() const;

    public:
        // Resampled braking profile on the fixed phase grid.
        struct BrakeProfile {
            double begin_time = 0.0;            // absolute start time of the
                                                 // window this profile was built
                                                 // from (localization's is the
                                                 // target's, shifted by the lag
                                                 // used for that update)
            std::vector<double> phase;          // size kPhaseCount
            std::vector<double> speed;          // size kPhaseCount
            std::vector<double> acceleration;   // size kPhaseCount
            std::vector<double> delta_integral; // size kPhaseCount - 1
            double duration = 0.0;              // real-time length of the window
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

            double lag = 0.0;            // lag applied for this update (seconds)
            double lag_confidence = 0.0; // cross-correlation score, [-1, 1];
                                          // equals the fallback current_lag_'s
                                          // last known confidence when this
                                          // event's own estimate wasn't used
        };

        std::optional<BrakeCompensationBuilder::DebugInfo> GetDebugInfo();

    private:
        struct Sample {
            double time = 0.0;
            double acc = 0.0;
            double velocity = 0.0;
        };

        // Streaming state of the target braking event.
        struct ActiveEvent {
            std::vector<Sample> samples;
            bool active = false;
        };

        // A target event that finished its monotonic deepening part but is not
        // yet matchable: the localization history needs to accumulate up to
        // peak_time + max_lag_ before every candidate lag is checkable.
        struct PendingTargetEvent {
            std::vector<Sample> smoothed_samples; // full buffered event, smoothed
            std::size_t peak_index = 0;           // end of the kept (monotonic)
                                                   // part within smoothed_samples
        };

        struct LagEstimate {
            double lag = 0.0;
            double confidence = -2.0; // Pearson r is in [-1, 1]; -2 == invalid
            bool valid = false;
        };

        static constexpr double kSamplePeriod = 0.02;
        static constexpr std::size_t kPhaseCount = 11; // phases 0.0 .. 1.0 step 0.1
        static constexpr double kEpsilon = 1e-9;

        void InitParams();

        // Feed a sample into the target's streaming event; on completion, queue
        // it as a PendingTargetEvent.
        void PushSample(ActiveEvent& event, const Sample& sample);

        // Analyse a finished (buffered) target event and, if usable, queue it.
        void CompleteEvent(ActiveEvent& event);

        // Zero-phase (centered) moving average of the samples' acceleration.
        std::vector<double> SmoothAcceleration(
            const std::vector<Sample>& samples) const;

        // First significant local minimum of `acc` using hysteresis (see class doc)
        // -- the index up to which the deepening part is kept.
        std::size_t FindPeakIndex(const std::vector<double>& acc) const;

        // Sliding-window median of a value series (spike-robust prefilter).
        std::vector<double> MedianFilter(const std::vector<double>& values) const;

        // Match and fold into the table every pending target event whose
        // required localization history (up to peak_time + max_lag_) has now
        // arrived; drop stale history no longer needed by any pending event.
        void ProcessPendingTargetEvents();

        // Estimate the lag between `target_event_samples` (a target event's
        // full smoothed buffer -- deepening plus release, so it has a real
        // minimum to anchor on) and localization_history_, by normalized
        // cross-correlation over candidate lags in [min_lag_, max_lag_].
        LagEstimate EstimateLag(const std::vector<Sample>& target_event_samples) const;

        // Copy of localization_history_ covering [begin_time, end_time], plus
        // one padding sample on each side (when available) for interpolation.
        std::vector<Sample> ExtractHistorySlice(
            double begin_time, double end_time) const;

        // Drop localization history no longer reachable by any pending event.
        void TrimLocalizationHistory();

        void MatchAndUpdate(const PendingTargetEvent& event);

        BrakeProfile BuildProfileFromSamples(
            const std::vector<Sample>& samples,
            double begin_time,
            double duration) const;

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
            const BrakeProfile& localization_profile,
            double lag,
            double lag_confidence);

        std::size_t FindNearestSpeedIndex(double speed) const;
        std::size_t FindNearestAccelerationIndex(double acceleration) const;

        double release_threshold_;
        double update_rate_;
        std::size_t smoothing_half_;      // effective window = 2 * half + 1
        double peak_release_margin_;
        std::size_t median_half_;         // effective window = 2 * half + 1
        double current_time_ = 0.0;

        ActiveEvent target_event_;
        std::deque<Sample> localization_history_;
        std::deque<PendingTargetEvent> pending_target_events_;

        BrakeCompensationParams params_;
        std::vector<double> update_counts_;
        std::optional<DebugInfo> debug_info_;
        double min_braking_duration_;

        double min_lag_;             // lower bound of the lag search range
        double max_lag_;             // upper bound of the lag search range, and
                                      // how long a completed event waits for
                                      // localization history before matching
        double lag_update_rate_;     // EMA rate for current_lag_
        double min_lag_correlation_; // confidence floor to trust a fresh
                                      // per-event lag estimate over current_lag_
        double current_lag_ = 0.0;   // learned target->localization lag (s)
        double current_lag_confidence_ = 0.0; // confidence of current_lag_'s
                                               // last accepted fresh estimate
    };

} // namespace yandex::sdc::control
