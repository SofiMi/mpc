#include "acceleration_matrix.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace yandex::sdc::control {

    namespace {

        bool IsFinite(double value) {
            return std::isfinite(value);
        }

        double Clamp(double value, double min_value, double max_value) {
            return std::max(min_value, std::min(value, max_value));
        }

        // Whether `samples` (sorted by time) has continuous real coverage
        // across [begin_time, end_time]: real data reaching within `max_gap`
        // of both edges, and no gap between consecutive samples wider than
        // `max_gap` anywhere in between. Templated (rather than named on
        // BrakeCompensationBuilder::Sample) only so this free function
        // doesn't need access to that private nested type.
        template <typename SampleT>
        bool HasContinuousCoverage(
            const std::vector<SampleT>& samples,
            double begin_time,
            double end_time,
            double max_gap) {
            if (samples.empty()) {
                return false;
            }
            if (samples.front().time > begin_time + max_gap) {
                return false;
            }
            if (samples.back().time < end_time - max_gap) {
                return false;
            }
            for (std::size_t i = 1; i < samples.size(); ++i) {
                if (samples[i].time - samples[i - 1].time > max_gap) {
                    return false;
                }
            }
            return true;
        }

    } // namespace

    StructInterpolant2d<ScalarAsArray<double>>
    BrakeCompensationParams::BuildInterpolant() const {
        const std::size_t n_speed = speed_points.size();
        const std::size_t n_acc = acceleration_points.size();
        std::vector<std::vector<ScalarAsArray<double>>> value_knots(
            n_acc, std::vector<ScalarAsArray<double>>(n_speed));
        for (std::size_t i = 0; i < n_acc; ++i) {
            for (std::size_t j = 0; j < n_speed; ++j) {
                value_knots[i][j] =
                    ScalarAsArray<double>{value_points[i * n_speed + j]};
            }
        }
        return StructInterpolant2d<ScalarAsArray<double>>{
            speed_points, acceleration_points, value_knots};
    }

    BrakeCompensationBuilder::BrakeCompensationBuilder(
        double release_threshold,
        double update_rate,
        std::size_t smoothing_window,
        double peak_release_margin,
        std::size_t median_window,
        double min_braking_duration,
        double min_lag,
        double max_lag,
        double lag_update_rate,
        double localization_entry_threshold)
        : release_threshold_(release_threshold)
        , update_rate_(Clamp(update_rate, 0.0, 1.0))
        , smoothing_half_(smoothing_window > 1 ? smoothing_window / 2 : 0)
        , peak_release_margin_(std::max(0.0, peak_release_margin))
        , median_half_(median_window > 1 ? median_window / 2 : 0)
        , min_braking_duration_(std::max(0.0, min_braking_duration))
        , min_lag_(std::max(0.0, min_lag))
        , max_lag_(std::max(min_lag_, max_lag))
        , lag_update_rate_(Clamp(lag_update_rate, 0.0, 1.0))
        , localization_entry_threshold_(localization_entry_threshold) {
        // Both thresholds are always used as negative values: braking starts
        // when acceleration drops to or below them.
        if (release_threshold_ >= 0.0) {
            release_threshold_ = -std::abs(release_threshold_);
        }
        if (localization_entry_threshold_ >= 0.0) {
            localization_entry_threshold_ = -std::abs(localization_entry_threshold_);
        }
        InitParams();
    }

    void BrakeCompensationBuilder::InitParams() {
        params_.speed_points.clear();
        params_.acceleration_points.clear();
        // Speed grid: 0, 1, ..., 10 m/s.
        for (int i = 0; i <= 10; ++i) {
            params_.speed_points.push_back(static_cast<double>(i));
        }
        // Acceleration grid: -4.0, -3.5, ..., -0.5 m/s^2.
        for (int i = 0; i <= 8; ++i) {
            params_.acceleration_points.push_back(-4.0 + 0.5 * i);
        }
        const std::size_t value_count =
            params_.speed_points.size() * params_.acceleration_points.size();
        params_.value_points.assign(value_count, 1.0);
        params_.enable = true;
        update_counts_.assign(value_count, 0.0);
    }

    void BrakeCompensationBuilder::Set(
        double localization_acc,
        double target_acc,
        double localization_vel) {
        const double time = current_time_;
        current_time_ += kSamplePeriod;

        if (IsFinite(localization_acc) && IsFinite(localization_vel)) {
            localization_history_.push_back(Sample{time, localization_acc, localization_vel});
        }

        Sample target_sample;
        target_sample.time = time;
        target_sample.acc = target_acc;
        target_sample.velocity = localization_vel;
        PushSample(target_event_, target_sample);

        ProcessPendingTargetEvents();
    }

    std::vector<double> BrakeCompensationBuilder::MedianFilter(
        const std::vector<double>& values) const {
        const std::size_t n = values.size();
        if (median_half_ == 0 || n == 0) {
            return values;
        }
        std::vector<double> out(n);
        const std::ptrdiff_t half = static_cast<std::ptrdiff_t>(median_half_);
        const std::ptrdiff_t last = static_cast<std::ptrdiff_t>(n) - 1;
        std::vector<double> window;
        window.reserve(2 * median_half_ + 1);
        for (std::ptrdiff_t i = 0; i <= last; ++i) {
            window.clear();
            for (std::ptrdiff_t k = -half; k <= half; ++k) {
                const std::ptrdiff_t j = i + k;
                if (j < 0 || j > last) {
                    continue;
                }
                window.push_back(values[static_cast<std::size_t>(j)]);
            }
            std::sort(window.begin(), window.end());
            out[static_cast<std::size_t>(i)] = window[window.size() / 2];
        }
        return out;
    }

    void BrakeCompensationBuilder::PushSample(
        ActiveEvent& event, const Sample& sample) {
        if (!IsFinite(sample.acc) || !IsFinite(sample.velocity)) {
            return;
        }

        if (!event.active) {
            // Wait for braking to cross the release threshold.
            if (sample.acc > -0.2) {
                return;
            }
            event.samples.clear();
            event.active = true;
            event.samples.push_back(sample);
            return;
        }

        // While active, buffer the raw sample. The event ends once acceleration
        // returns to zero or above; the peak is then chosen from the whole buffer.
        event.samples.push_back(sample);
        if (sample.acc >= -0.2) {
            CompleteEvent(event);
        }
    }

    void BrakeCompensationBuilder::CompleteEvent(ActiveEvent& event) {
        const std::vector<Sample> samples = std::move(event.samples);
        event = ActiveEvent{};

        if (samples.size() < 2) {
            return;
        }

        // Smooth first (zero-phase, no lag -- the full event is available), then
        // find the peak of the trend with hysteresis so that noise cannot cut the
        // deepening part short.
        const std::vector<double> smoothed = SmoothAcceleration(samples);
        const std::size_t peak_index = FindPeakIndex(smoothed);
        if (peak_index < 1) {
            // No real deepening segment (single spike / noise).
            return;
        }

        if (smoothed[peak_index] > release_threshold_) {
            // The braking never got deep enough: its peak stayed above the release
            // threshold, so the whole maneuver is treated as insignificant and does
            // not update the table (even though it was collected from ~0).
            return;
        }

        // Keep only the kept (monotonic deepening) part, using the de-noised
        // acceleration so the integrals reflect the trend rather than the
        // rectified noise.
        std::vector<Sample> kept(
            samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(peak_index) + 1);
        for (std::size_t i = 0; i < kept.size(); ++i) {
            kept[i].acc = smoothed[i];
        }

        const double duration = kept.back().time - kept.front().time;
        if (duration < min_braking_duration_) {
            return;
        }

        pending_target_events_.push_back(PendingTargetEvent{std::move(kept)});
    }

    std::vector<double> BrakeCompensationBuilder::SmoothAcceleration(
        const std::vector<Sample>& samples) const {
        const std::size_t n = samples.size();
        std::vector<double> acc(n);
        for (std::size_t i = 0; i < n; ++i) {
            acc[i] = samples[i].acc;
        }

        // Median prefilter first: it discards short outliers (single/double-sample
        // spikes) outright, so a spurious deep spike can neither become the detected
        // peak nor inflate the integral. A moving average then smooths the residual
        // jitter of the trend.
        acc = MedianFilter(acc);

        if (smoothing_half_ == 0) {
            return acc;
        }
        std::vector<double> out(n);
        const std::ptrdiff_t half = static_cast<std::ptrdiff_t>(smoothing_half_);
        const std::ptrdiff_t last = static_cast<std::ptrdiff_t>(n) - 1;
        for (std::ptrdiff_t i = 0; i <= last; ++i) {
            double sum = 0.0;
            std::size_t count = 0;
            for (std::ptrdiff_t k = -half; k <= half; ++k) {
                const std::ptrdiff_t j = i + k;
                if (j < 0 || j > last) {
                    continue;
                }
                sum += acc[static_cast<std::size_t>(j)];
                ++count;
            }
            out[static_cast<std::size_t>(i)] =
                count > 0 ? sum / static_cast<double>(count)
                        : acc[static_cast<std::size_t>(i)];
        }
        return out;
    }

    std::size_t BrakeCompensationBuilder::FindPeakIndex(
        const std::vector<double>& acc) const {
        if (acc.empty()) {
            return 0;
        }
        std::size_t min_index = 0;
        double min_value = acc[0];
        for (std::size_t i = 1; i < acc.size(); ++i) {
            if (acc[i] < min_value - kEpsilon) {
                // Still deepening (or a new global minimum of the trend).
                min_value = acc[i];
                min_index = i;
            } else if (acc[i] > min_value + peak_release_margin_) {
                // A rise larger than the noise band: the deepening part is over.
                return min_index;
            }
            // Anything in between is treated as noise around the running minimum.
        }
        return min_index;
    }

    void BrakeCompensationBuilder::ProcessPendingTargetEvents() {
        while (!pending_target_events_.empty()) {
            const PendingTargetEvent& pending = pending_target_events_.front();
            const double peak_time = pending.kept_samples.back().time;
            if (current_time_ < peak_time + max_lag_) {
                // Localization hasn't caught up far enough yet to search the
                // full [min_lag_, max_lag_] range for this event; try again on
                // a later Set() call once more history has accumulated.
                break;
            }
            PendingTargetEvent event = std::move(pending_target_events_.front());
            pending_target_events_.pop_front();
            MatchAndUpdate(event);
        }
        TrimLocalizationHistory();
    }

    void BrakeCompensationBuilder::TrimLocalizationHistory() {
        const double floor_time = pending_target_events_.empty()
            ? current_time_ - max_lag_
            : pending_target_events_.front().kept_samples.front().time;
        while (!localization_history_.empty() &&
               localization_history_.front().time < floor_time - kSamplePeriod) {
            localization_history_.pop_front();
        }
    }

    std::vector<BrakeCompensationBuilder::Sample>
    BrakeCompensationBuilder::ExtractHistorySlice(
        double begin_time, double end_time) const {
        std::vector<Sample> slice;
        if (localization_history_.empty() || end_time < begin_time) {
            return slice;
        }
        auto first = std::lower_bound(
            localization_history_.begin(),
            localization_history_.end(),
            begin_time,
            [](const Sample& sample, double value) { return sample.time < value; });
        if (first != localization_history_.begin()) {
            --first; // one padding sample before the window, for interpolation
        }
        auto last = std::upper_bound(
            localization_history_.begin(),
            localization_history_.end(),
            end_time,
            [](double value, const Sample& sample) { return value < sample.time; });
        if (last != localization_history_.end()) {
            ++last; // one padding sample after the window
        }
        slice.assign(first, last);
        return slice;
    }

    BrakeCompensationBuilder::LagEstimate BrakeCompensationBuilder::EstimateLag(
        double begin_time) const {
        LagEstimate result;

        const double search_begin = begin_time + min_lag_;
        const double search_end = begin_time + max_lag_;

        // Same idea as the target's own entry-threshold detection in
        // PushSample, just applied to localization and scoped to search only
        // in [search_begin, search_end] for this specific target event --
        // never at localization_history_'s absolute start, which could belong
        // to an unrelated, older maneuver.
        auto it = std::lower_bound(
            localization_history_.begin(),
            localization_history_.end(),
            search_begin,
            [](const Sample& sample, double value) { return sample.time < value; });
        for (; it != localization_history_.end() && it->time <= search_end; ++it) {
            if (it->acc <= localization_entry_threshold_) {
                result.lag = it->time - begin_time;
                result.valid = true;
                return result;
            }
        }
        return result;
    }

    void BrakeCompensationBuilder::MatchAndUpdate(const PendingTargetEvent& event) {
        const BrakeProfile target_profile = BuildProfileFromSamples(
            event.kept_samples,
            event.kept_samples.front().time,
            event.kept_samples.back().time - event.kept_samples.front().time);
        if (target_profile.phase.empty()) {
            return;
        }

        const LagEstimate estimate = EstimateLag(target_profile.begin_time);

        // Prefer this event's own localization entry crossing when found;
        // otherwise fall back to the last learned lag rather than guessing.
        double lag = current_lag_;
        bool fresh_estimate_used = false;
        if (estimate.valid) {
            lag = estimate.lag;
            fresh_estimate_used = true;
        }

        const double window_begin = target_profile.begin_time + lag;
        const double window_end = window_begin + target_profile.duration;
        const std::vector<Sample> localization_slice =
            ExtractHistorySlice(window_begin, window_end);

        // The window needs continuous real localization coverage before it's
        // even built: otherwise Interpolate would clamp to (or bridge
        // across) whatever real sample is nearest, however far away -- which
        // can still be deep enough to pass the plain magnitude check below,
        // since it's a real sample, just from an unrelated moment (this is
        // exactly how a match on a mostly-empty window used to sneak
        // through: the whole window read back as one repeated, distant
        // sample that happened to be deep).
        if (!HasContinuousCoverage(localization_slice, window_begin, window_end, kMaxHistoryGap)) {
            return;
        }

        // Smooth localization the same way target already was in
        // CompleteEvent (median prefilter, then moving average): raw history
        // is exactly as noisy as target's raw buffer, and comparing a
        // smoothed target against raw localization let a single noisy dip in
        // localization (toward zero, or spiking) skew that phase point's
        // contribution to its table cell far more than the real trend
        // should.
        std::vector<Sample> smoothed_localization_slice = localization_slice;
        const std::vector<double> smoothed_localization_values =
            SmoothAcceleration(localization_slice);
        for (std::size_t i = 0; i < smoothed_localization_slice.size(); ++i) {
            smoothed_localization_slice[i].acc = smoothed_localization_values[i];
        }

        const BrakeProfile localization_profile = BuildProfileFromSamples(
            smoothed_localization_slice, window_begin, target_profile.duration);
        if (localization_profile.phase.empty()) {
            return;
        }

        // The shifted localization window must show real braking -- otherwise
        // this match (fresh estimate or fallback to current_lag_) isn't
        // trustworthy enough to fold into the table.
        const double localization_min = *std::min_element(
            localization_profile.acceleration.begin(),
            localization_profile.acceleration.end());
        if (localization_min > release_threshold_) {
            return;
        }

        if (fresh_estimate_used) {
            current_lag_ =
                (1.0 - lag_update_rate_) * current_lag_ + lag_update_rate_ * lag;
        }

        UpdateParams(target_profile, localization_profile, lag, fresh_estimate_used);
    }

    BrakeCompensationBuilder::BrakeProfile
    BrakeCompensationBuilder::BuildProfileFromSamples(
        const std::vector<Sample>& samples,
        double begin_time,
        double duration) const {
        BrakeProfile profile;
        if (samples.size() < 2 || duration <= kEpsilon) {
            return profile;
        }

        profile.begin_time = begin_time;
        profile.duration = duration;
        profile.phase.resize(kPhaseCount);
        profile.speed.resize(kPhaseCount);
        profile.acceleration.resize(kPhaseCount);
        profile.delta_integral.resize(kPhaseCount - 1);

        for (std::size_t i = 0; i < kPhaseCount; ++i) {
            const double phase = static_cast<double>(i) / (kPhaseCount - 1);
            const double time = begin_time + phase * duration;
            profile.phase[i] = phase;
            profile.speed[i] = Interpolate(samples, time, /*velocity=*/true);
            profile.acceleration[i] = Interpolate(samples, time, /*velocity=*/false);
        }

        for (std::size_t i = 0; i + 1 < kPhaseCount; ++i) {
            const double t0 = begin_time + profile.phase[i] * duration;
            const double t1 = begin_time + profile.phase[i + 1] * duration;
            profile.delta_integral[i] = IntegrateAbsAcceleration(samples, t0, t1);
        }

        profile.acceleration_segments.resize(kPhaseCount - 1);
        profile.time_segments.resize(kPhaseCount - 1);

        std::vector<double> boundaries(kPhaseCount);
        for (std::size_t i = 0; i < kPhaseCount; ++i) {
            boundaries[i] = begin_time + profile.phase[i] * duration;
        }

        for (const auto& sample : samples) {
            // Padding samples from ExtractHistorySlice can fall just outside the
            // window; they exist only so Interpolate has real neighbors at the
            // boundaries and shouldn't be attributed to a segment.
            if (sample.time < boundaries.front() || sample.time > boundaries.back()) {
                continue;
            }
            auto it = std::upper_bound(boundaries.begin(), boundaries.end(), sample.time);
            std::size_t interval;
            if (it == boundaries.begin()) {
                interval = 0;
            } else if (it == boundaries.end()) {
                interval = boundaries.size() - 2;
            } else {
                interval = static_cast<std::size_t>(it - boundaries.begin() - 1);
            }
            profile.acceleration_segments[interval].push_back(sample.acc);
            profile.time_segments[interval].push_back(sample.time);
        }

        return profile;
    }

    double BrakeCompensationBuilder::Interpolate(
        const std::vector<Sample>& samples,
        double time,
        bool velocity) const {
        if (samples.empty()) {
            return 0.0;
        }
        if (samples.size() == 1 || time <= samples.front().time) {
            return velocity ? samples.front().velocity : samples.front().acc;
        }
        if (time >= samples.back().time) {
            return velocity ? samples.back().velocity : samples.back().acc;
        }

        auto upper = std::upper_bound(
            samples.begin(),
            samples.end(),
            time,
            [](double value, const Sample& sample) { return value < sample.time; });
        const Sample& right = *upper;
        const Sample& left = *(upper - 1);
        const double dt = right.time - left.time;
        if (dt <= kEpsilon) {
            return velocity ? right.velocity : right.acc;
        }
        const double ratio = (time - left.time) / dt;
        const double left_value = velocity ? left.velocity : left.acc;
        const double right_value = velocity ? right.velocity : right.acc;
        return left_value + (right_value - left_value) * ratio;
    }

    double BrakeCompensationBuilder::IntegrateAbsAcceleration(
        const std::vector<Sample>& samples,
        double begin_time,
        double end_time) const {
        if (samples.empty() || end_time <= begin_time) {
            return 0.0;
        }

        // Integrate over the native sample grid (step kSamplePeriod), splitting the
        // partial segments at the phase boundaries. Acceleration at the boundaries
        // is obtained by linear interpolation.
        std::vector<double> times;
        times.reserve(samples.size() + 2);
        times.push_back(begin_time);
        for (const auto& sample : samples) {
            if (sample.time > begin_time + kEpsilon &&
                sample.time < end_time - kEpsilon) {
                times.push_back(sample.time);
            }
        }
        times.push_back(end_time);

        double integral = 0.0;
        for (std::size_t i = 1; i < times.size(); ++i) {
            const double t0 = times[i - 1];
            const double t1 = times[i];
            const double a0 = Interpolate(samples, t0, /*velocity=*/false);
            const double a1 = Interpolate(samples, t1, /*velocity=*/false);
            // Braking acceleration is negative; accumulate the magnitude.
            integral += 0.5 * (std::abs(a0) + std::abs(a1)) * (t1 - t0);
        }
        return integral;
    }

    std::size_t BrakeCompensationBuilder::FindNearestSpeedIndex(
        double speed) const {
        if (params_.speed_points.empty()) {
            return 0;
        }
        std::size_t best_index = 0;
        double best_distance = std::abs(speed - params_.speed_points.front());
        for (std::size_t i = 1; i < params_.speed_points.size(); ++i) {
            const double distance = std::abs(speed - params_.speed_points[i]);
            if (distance < best_distance) {
                best_distance = distance;
                best_index = i;
            }
        }
        return best_index;
    }

    std::size_t BrakeCompensationBuilder::FindNearestAccelerationIndex(
        double acceleration) const {
        if (params_.acceleration_points.empty()) {
            return 0;
        }
        std::size_t best_index = 0;
        double best_distance =
            std::abs(acceleration - params_.acceleration_points.front());
        for (std::size_t i = 1; i < params_.acceleration_points.size(); ++i) {
            const double distance =
                std::abs(acceleration - params_.acceleration_points[i]);
            if (distance < best_distance) {
                best_distance = distance;
                best_index = i;
            }
        }
        return best_index;
    }

    void BrakeCompensationBuilder::UpdateParams(
        const BrakeProfile& target_profile,
        const BrakeProfile& localization_profile,
        double lag,
        bool lag_is_fresh) {
        const std::size_t point_count = std::min(
            target_profile.acceleration.size(),
            localization_profile.acceleration.size());
        if (point_count == 0 || update_rate_ <= 0.0) {
            return;
        }
        const std::size_t n_speed = params_.speed_points.size();
        if (n_speed == 0 || params_.acceleration_points.empty()) {
            return;
        }

        std::vector<double> vel;
        std::vector<double> acc;
        std::vector<double> coef_old;
        std::vector<double> coef_new;
        std::vector<double> coef_res;

        // Коэффициент = отношение ускорений на совпадающей фазе,
        // |a_target(phase)| / |a_loc(phase)| -- насколько сильнее тормозит таргет,
        // чем локализация, в той же точке манёвра. Локализация запаздывает и её
        // торможение длится дольше -- то есть на той же фазе её ускорение СЛАБЕЕ;
        // именно это отношение и должно поймать. Отношение сырых интегралов по
        // времени домножило бы результат на (бОльшую) длительность локализации ещё
        // раз и увело бы коэффициент обратно к 1. Фазовая сетка равномерна по
        // времени, поэтому фазовые отсчёты ускорения -- это уже среднее,
        // нормированное на длительность; сравниваем их напрямую.
        struct CellAccumulator {
            double target = 0.0;
            double localization = 0.0;
            double acc = 0.0;
            double vel = 0.0;
        };
        std::unordered_map<std::size_t, CellAccumulator> per_cell;
        per_cell.reserve(point_count);

        for (std::size_t i = 0; i < point_count; ++i) {
            const double target_acceleration = target_profile.acceleration[i];
            const double localization_acceleration =
                localization_profile.acceleration[i];
            if (!IsFinite(target_acceleration) ||
                !IsFinite(localization_acceleration)) {
                continue;
            }

            // Ключ: скорость из потока локализации, ускорение из потока таргета.
            const double speed = localization_profile.speed[i];
            const std::size_t speed_index = FindNearestSpeedIndex(speed);
            const std::size_t acceleration_index =
                FindNearestAccelerationIndex(target_acceleration);
            const std::size_t value_index = acceleration_index * n_speed + speed_index;
            if (value_index >= params_.value_points.size()) {
                continue;
            }

            CellAccumulator& cell = per_cell[value_index];
            cell.target += std::abs(target_acceleration);
            cell.localization += std::abs(localization_acceleration);
            cell.acc = target_acceleration;
            cell.vel = speed;
        }

        // Один шаг EMA на ячейку от схлопнутого коэффициента.
        for (const auto& [value_index, cell] : per_cell) {
            if (cell.localization <= kEpsilon) {
                continue;
            }
            const double coefficient = cell.target / cell.localization;
            if (!IsFinite(coefficient) || coefficient <= 0.0) {
                continue;
            }
            double& count = update_counts_[value_index];
            count += 1.0;
            const double rate = update_rate_;
            const double old_value = params_.value_points[value_index];
            params_.value_points[value_index] = std::clamp((1.0 - rate) * old_value + rate * coefficient, 1.0, 1.5);

            vel.push_back(cell.vel);
            acc.push_back(cell.acc);
            coef_old.push_back(old_value);
            coef_new.push_back(coefficient);
            coef_res.push_back(params_.value_points[value_index]);
        }

        if (vel.size() > 0) {
            BrakeCompensationBuilder::DebugInfo debug_info;
            debug_info.target = target_profile;
            debug_info.current = localization_profile;
            debug_info.vel = vel;
            debug_info.acc = acc;
            debug_info.coef_old = coef_old;
            debug_info.coef_new = coef_new;
            debug_info.coef_res = coef_res;
            debug_info.lag = lag;
            debug_info.lag_is_fresh = lag_is_fresh;
            debug_info_ = debug_info;
        }
    }

    void BrakeCompensationBuilder::Clear() {
        target_event_ = ActiveEvent{};
        localization_history_.clear();
        pending_target_events_.clear();
        current_time_ = 0.0;
        current_lag_ = 0.0;
        // Per the class doc, Clear() also resets the table itself, not just the
        // in-flight event/history/lag state.
        InitParams();
    }

    BrakeCompensationParams BrakeCompensationBuilder::GetParams() const {
        return params_;
    }

    std::optional<BrakeCompensationBuilder::DebugInfo> BrakeCompensationBuilder::GetDebugInfo() {
        return debug_info_;
    }

} // namespace yandex::sdc::control
