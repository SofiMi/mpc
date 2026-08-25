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
    //     on the first sample with acceleration <= -0.2 and ends once
    //     acceleration returns to zero or above. Only the monotonically
    //     deepening part, from the start to the first local minimum (the peak
    //     of braking), is kept.
    //   * Earlier designs tried to line target and localization up by
    //     normalized phase (independent onset detection on both streams,
    //     matched afterwards) and then by an explicitly estimated time lag
    //     (a cross-correlation search, then a simpler entry-threshold scan on
    //     localization). Both kept finding new ways to be fooled: onset
    //     noise distorting the phase-0 point, Interpolate clamping across a
    //     stretch of history with no real data and looking like a match, a
    //     lag search settling for a low-confidence guess when the true lag
    //     was out of range. Comparing at the same absolute time was tried too
    //     and rejected for ignoring the real lag entirely.
    //   * The current approach: within target's kept (monotonic) part, find
    //     where its smoothed acceleration first reaches sync_threshold_ (a
    //     deeper, less noise-prone point than the -0.2 activation) and start
    //     the comparison window there, running to the peak. Localization is
    //     read at those exact same absolute times -- no lag, no search. The
    //     window must have continuous real localization coverage (see
    //     kMaxHistoryGap) before it's even built: otherwise Interpolate would
    //     clamp to (or bridge across) whatever real sample is nearest,
    //     however far away, which can be deep enough to pass a plain
    //     magnitude check despite belonging to an unrelated moment. A covered
    //     window is smoothed the same way target's own kept samples were
    //     (median prefilter, then moving average): raw localization history
    //     is exactly as noisy as target's raw buffer, and comparing a
    //     smoothed target against raw localization let a single noisy dip
    //     skew that phase point's contribution to its table cell. The
    //     smoothed window is then still sanity-checked (it must dip to
    //     release_threshold_ or below) before the pair is used, so a bad
    //     match cannot corrupt the table.
    //   * Because target's own event only completes once its release phase
    //     crosses back above -0.2 -- strictly after the peak the comparison
    //     window ends at -- localization_history_ already covers the whole
    //     window by the time the event completes. Matching happens
    //     synchronously right there: no waiting, no pending-event queue.
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
            double sync_threshold = -0.3);

        // Feed one synchronized measurement. `localization_vel` is the current
        // speed, used both as the profile speed and as the table's speed key.
        void Set(
            double localization_acc,
            double target_acc,
            double localization_vel);

        // Drop all in-flight state and reset the table to its initial
        // (all-ones) grid.
        void Clear();

        // Current accumulated table.
        BrakeCompensationParams GetParams() const;

    public:
        // Resampled braking profile on the fixed phase grid.
        struct BrakeProfile {
            double begin_time = 0.0;            // absolute start time of the
                                                 // comparison window (same for
                                                 // target and localization)
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

        // Feed a sample into the target's streaming event; on completion,
        // analyse and (if usable) match it against localization immediately.
        void PushSample(ActiveEvent& event, const Sample& sample);

        // Analyse a finished (buffered) target event and, if usable, match it.
        void CompleteEvent(ActiveEvent& event);

        // Zero-phase (centered) moving average of the samples' acceleration.
        std::vector<double> SmoothAcceleration(
            const std::vector<Sample>& samples) const;

        // First significant local minimum of `acc` using hysteresis (see class doc)
        // -- the index up to which the deepening part is kept.
        std::size_t FindPeakIndex(const std::vector<double>& acc) const;

        // Sliding-window median of a value series (spike-robust prefilter).
        std::vector<double> MedianFilter(const std::vector<double>& values) const;

        // Copy of localization_history_ covering [begin_time, end_time], plus
        // one padding sample on each side (when available) for interpolation.
        std::vector<Sample> ExtractHistorySlice(
            double begin_time, double end_time) const;

        // Drop localization history no longer reachable by the active target
        // event (if any).
        void TrimLocalizationHistory();

        // Build both profiles over `kept` (target's window, starting at its
        // sync_threshold_ crossing) and, if localization checks out, fold the
        // pair into the table.
        void MatchAndUpdate(const std::vector<Sample>& kept);

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
        std::deque<Sample> localization_history_;

        BrakeCompensationParams params_;
        std::vector<double> update_counts_;
        std::optional<DebugInfo> debug_info_;
        double min_braking_duration_;
        double sync_threshold_; // stored negative; see class doc
    };

} // namespace yandex::sdc::control
