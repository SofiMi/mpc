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
    //     detected as its own event -- its raw samples are just kept in a
    //     rolling history buffer (localization_history_). Detecting braking
    //     independently on both streams and matching by normalized phase used
    //     to be how this worked, but that conflates the real physical lag
    //     between the streams with each stream's own onset-detection noise (a
    //     noisier signal crosses its own threshold later and already deeper
    //     than it should be), which silently distorted the phase-0 point. A
    //     cross-correlation-based lag search was tried next, but proved too
    //     easy to fool: Interpolate can't distinguish "real data here" from
    //     "clamped to the nearest edge sample of a stretch with nothing in
    //     it", so a sparse/gappy localization history could still score a
    //     passable correlation at a bogus lag.
    //   * Instead, once a target event's kept (monotonic) window
    //     [begin_time, peak_time] is known, localization_history_ is scanned
    //     forward from begin_time + min_lag_ (up to begin_time + max_lag_) for
    //     the first sample at or below localization_entry_threshold_ -- the
    //     same kind of simple entry-threshold crossing already used to detect
    //     the target's own event, just applied to localization and scoped to
    //     search only near this specific target event. That crossing's offset
    //     from begin_time is the lag. A found lag updates current_lag_ (an
    //     EMA, so a car's actual brake/estimation lag is learned over time
    //     rather than assumed); if nothing crosses the threshold in range, the
    //     last learned current_lag_ is used as a fallback instead of guessing.
    //     Either way, the resulting localization window
    //     [begin_time + lag, peak_time + lag] must have continuous real
    //     localization coverage (see kMaxHistoryGap) before it's even built:
    //     otherwise Interpolate would clamp to (or bridge across) whatever
    //     real sample is nearest, however far away, which can be deep enough
    //     to pass a plain magnitude check despite belonging to an unrelated
    //     moment. A covered window is smoothed the same way target's own kept
    //     samples were in CompleteEvent (median prefilter, then moving
    //     average) before its profile is built -- raw localization history is
    //     exactly as noisy as target's raw buffer, and comparing a smoothed
    //     target against raw localization let a single noisy dip skew that
    //     phase point's contribution to its table cell. The smoothed window is
    //     then still sanity-checked (it must dip to release_threshold_ or
    //     below) before the pair is used, so a bad match cannot corrupt the
    //     table.
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
            double localization_entry_threshold = -0.3);

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

            double lag = 0.0;         // lag applied for this update (seconds)
            bool lag_is_fresh = false; // true if this event's own localization
                                        // entry crossing was found; false if it
                                        // fell back to the last learned
                                        // current_lag_
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
        // peak_time + max_lag_ before the whole lag search window is available.
        struct PendingTargetEvent {
            std::vector<Sample> kept_samples; // the kept (monotonic) part,
                                               // smoothed
        };

        struct LagEstimate {
            double lag = 0.0;
            bool valid = false;
        };

        static constexpr double kSamplePeriod = 0.02;
        static constexpr std::size_t kPhaseCount = 11; // phases 0.0 .. 1.0 step 0.1
        static constexpr double kEpsilon = 1e-9;
        // Max allowed gap, anywhere in [begin_time, end_time], between the
        // window's edges and the nearest real localization sample, and
        // between consecutive real samples, for that window to be considered
        // genuinely covered (see MatchAndUpdate). A few sample periods
        // tolerates the occasional dropped (non-finite) sample without
        // accepting a window that Interpolate would mostly fill in by
        // clamping to (or bridging across) a stretch with no real data --
        // which can still look "deep enough" to pass the plain magnitude
        // check below, since the clamped value is a real sample, just from
        // an unrelated moment.
        static constexpr double kMaxHistoryGap = 3 * kSamplePeriod;

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

        // Find the lag between a target event starting at `begin_time` and
        // localization: the offset (within [min_lag_, max_lag_]) of the first
        // localization_history_ sample at or below
        // localization_entry_threshold_.
        LagEstimate EstimateLag(double begin_time) const;

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
            bool lag_is_fresh);

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
        double localization_entry_threshold_; // localization's own onset
                                               // threshold (stored negative)
        double current_lag_ = 0.0;   // learned target->localization lag (s)
    };

} // namespace yandex::sdc::control
