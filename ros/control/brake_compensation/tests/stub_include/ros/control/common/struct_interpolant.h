#pragma once

// Minimal test-only stand-in for the real ros/control/common/struct_interpolant.h
// (not present in this checkout). It reproduces just enough of the interface
// that acceleration_matrix.{h,cpp} needs to compile: a 2D grid interpolant
// template and a scalar-wrapped-as-array value type. It does not implement
// actual interpolation and must never be used outside of this test build --
// swap in the real header from the production monorepo for anything else.

#include <utility>
#include <vector>

namespace yandex::sdc::control {

    template <typename T>
    struct ScalarAsArray {
        T value{};
    };

    template <typename T>
    class StructInterpolant2d {
    public:
        StructInterpolant2d() = default;

        StructInterpolant2d(
            std::vector<double> x_points,
            std::vector<double> y_points,
            std::vector<std::vector<T>> values)
            : x_points_(std::move(x_points))
            , y_points_(std::move(y_points))
            , values_(std::move(values)) {}

        const std::vector<double>& x_points() const {
            return x_points_;
        }

        const std::vector<double>& y_points() const {
            return y_points_;
        }

        const std::vector<std::vector<T>>& values() const {
            return values_;
        }

    private:
        std::vector<double> x_points_;
        std::vector<double> y_points_;
        std::vector<std::vector<T>> values_;
    };

} // namespace yandex::sdc::control
