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
        double max_event_time_offset,
        double max_duration_ratio)
        : release_threshold_(release_threshold)
        , update_rate_(Clamp(update_rate, 0.0, 1.0))
        , smoothing_half_(smoothing_window > 1 ? smoothing_window / 2 : 0)
        , peak_release_margin_(std::max(0.0, peak_release_margin))
        , median_half_(median_window > 1 ? median_window / 2 : 0)
        , min_braking_duration_(std::max(0.0, min_braking_duration))
        , max_event_time_offset_(std::max(0.0, max_event_time_offset))
        , max_duration_ratio_(std::max(1.0, max_duration_ratio)) {
        // The threshold is always used as a negative value: braking starts when
        // acceleration drops to or below it.
        if (release_threshold_ >= 0.0) {
            release_threshold_ = -std::abs(release_threshold_);
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

        Sample target_sample;
        target_sample.time = time;
        target_sample.acc = target_acc;
        target_sample.velocity = localization_vel;

        Sample localization_sample;
        localization_sample.time = time;
        localization_sample.acc = localization_acc;
        localization_sample.velocity = localization_vel;

        PushSample(target_event_, target_sample, target_profiles_);
        PushSample(localization_event_, localization_sample, localization_profiles_);
        TryUpdateParams();
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
        ActiveEvent& event,
        const Sample& sample,
        std::deque<BrakeProfile>& completed_profiles) {
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
            CompleteEvent(event, completed_profiles);
        }
    }

    void BrakeCompensationBuilder::CompleteEvent(
        ActiveEvent& event,
        std::deque<BrakeProfile>& completed_profiles) {
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

        BrakeEvent brake_event;
        brake_event.samples.assign(
            samples.begin(),
            samples.begin() + static_cast<std::ptrdiff_t>(peak_index) + 1);
        // Use the smoothed (de-noised) acceleration for the kept part so the
        // integrals reflect the trend rather than the rectified noise.
        for (std::size_t i = 0; i < brake_event.samples.size(); ++i) {
            brake_event.samples[i].acc = smoothed[i];
        }

        BrakeProfile profile = BuildProfile(brake_event);

        if (profile.duration < min_braking_duration_) {
            return;
        }

        if (!profile.delta_integral.empty()) {
            completed_profiles.push_back(std::move(profile));
        }
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

    void BrakeCompensationBuilder::TryUpdateParams() {
        while (!target_profiles_.empty() && !localization_profiles_.empty()) {
            // The two streams are filtered independently (peak validity, minimum
            // duration, ...), so either one can drop an event the other kept; a
            // plain FIFO pairing would then silently match unrelated maneuvers and
            // never recover. Guard the pairing by how far apart the two profiles'
            // start times are: a gap larger than the known localization lag means
            // the older profile's real counterpart never arrived, so it is
            // discarded on its own rather than paired with a later, unrelated one.
            const double offset = target_profiles_.front().begin_time -
                localization_profiles_.front().begin_time;
            if (offset > max_event_time_offset_) {
                localization_profiles_.pop_front();
                continue;
            }
            if (-offset > max_event_time_offset_) {
                target_profiles_.pop_front();
                continue;
            }

            BrakeProfile target_profile = std::move(target_profiles_.front());
            BrakeProfile localization_profile =
                std::move(localization_profiles_.front());
            target_profiles_.pop_front();
            localization_profiles_.pop_front();

            // A matched pair whose durations are grossly disproportionate is not a
            // reliable comparison: the shorter profile is just a handful of raw
            // samples stretched onto the fixed kPhaseCount grid, so its per-phase
            // values are dominated by interpolation/noise rather than the real
            // maneuver. Drop the pair instead of folding a distorted coefficient
            // into the table.
            const double shorter_duration =
                std::min(target_profile.duration, localization_profile.duration);
            const double longer_duration =
                std::max(target_profile.duration, localization_profile.duration);
            if (shorter_duration <= kEpsilon ||
                longer_duration / shorter_duration > max_duration_ratio_) {
                continue;
            }

            UpdateParams(target_profile, localization_profile);
        }
    }

    BrakeCompensationBuilder::BrakeProfile
    BrakeCompensationBuilder::BuildProfile(const BrakeEvent& event) const {
        BrakeProfile profile;
        if (event.samples.size() < 2) {
            return profile;
        }

        const double begin_time = event.samples.front().time;
        const double end_time = event.samples.back().time;
        const double duration = end_time - begin_time;
        if (duration <= kEpsilon) {
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
            profile.speed[i] = Interpolate(event.samples, time, /*velocity=*/true);
            profile.acceleration[i] =
                Interpolate(event.samples, time, /*velocity=*/false);
        }

        for (std::size_t i = 0; i + 1 < kPhaseCount; ++i) {
            const double t0 = begin_time + profile.phase[i] * duration;
            const double t1 = begin_time + profile.phase[i + 1] * duration;
            profile.delta_integral[i] =
                IntegrateAbsAcceleration(event.samples, t0, t1);
        }

        profile.acceleration_segments.resize(kPhaseCount - 1);

        // Времена границ интервалов
        std::vector<double> boundaries(kPhaseCount);
        for (std::size_t i = 0; i < kPhaseCount; ++i) {
            boundaries[i] = begin_time + profile.phase[i] * duration;
        }

        for (const auto& sample : event.samples) {
            // Найти интервал, в котором находится sample.time
            // Считаем, что сэмплы с временем вне [begin_time, end_time] не попадают (но они внутри, т.к. samples обрезаны до peak_index)
            // Используем upper_bound для границ
            auto it = std::upper_bound(boundaries.begin(), boundaries.end(), sample.time);
            if (it == boundaries.begin() || it == boundaries.end()) {
                // Сэмпл на границе или вне диапазона; можно отнести к ближайшему интервалу
                // Проще: если время == begin_time, отнести к первому интервалу; если == end_time, к последнему.
                if (sample.time <= boundaries.front()) {
                    profile.acceleration_segments[0].push_back(sample.acc);
                } else if (sample.time >= boundaries.back()) {
                    profile.acceleration_segments.back().push_back(sample.acc);
                }
                continue;
            }
            // it указывает на первый элемент > sample.time, значит интервал = it - boundaries.begin() - 1
            std::size_t interval = static_cast<std::size_t>(it - boundaries.begin() - 1);
            profile.acceleration_segments[interval].push_back(sample.acc);
        }

        profile.time_segments.resize(kPhaseCount - 1);
        for (const auto& sample : event.samples) {
        auto it = std::upper_bound(boundaries.begin(), boundaries.end(), sample.time);
            if (it == boundaries.begin() || it == boundaries.end()) {
                if (sample.time <= boundaries.front()) {
                    profile.acceleration_segments[0].push_back(sample.acc);
                    profile.time_segments[0].push_back(sample.time);  // <-- время
                } else if (sample.time >= boundaries.back()) {
                    profile.acceleration_segments.back().push_back(sample.acc);
                    profile.time_segments.back().push_back(sample.time);  // <-- время
                }
                continue;
            }
            std::size_t interval = static_cast<std::size_t>(it - boundaries.begin() - 1);
            profile.acceleration_segments[interval].push_back(sample.acc);
            profile.time_segments[interval].push_back(sample.time);  // <-- время
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
        const BrakeProfile& localization_profile) {
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
            debug_info_ = debug_info;
        }
    }

    void BrakeCompensationBuilder::Clear() {
        target_event_ = ActiveEvent{};
        localization_event_ = ActiveEvent{};
        target_profiles_.clear();
        localization_profiles_.clear();
        current_time_ = 0.0;
    }

    BrakeCompensationParams BrakeCompensationBuilder::GetParams() const {
        return params_;
    }

    std::optional<BrakeCompensationBuilder::DebugInfo> BrakeCompensationBuilder::GetDebugInfo() {
        return debug_info_;
    }

} // namespace yandex::sdc::control
