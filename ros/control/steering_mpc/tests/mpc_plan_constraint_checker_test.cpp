#include <gtest/gtest.h>

#include "sdg/sdc/ros/control/steering_mpc/mpc_plan_constraint_checker.h"

#include <cmath>
#include <vector>

namespace {
    using yandex::sdc::control::steering_mpc::MpcPlanConstraintChecker;
    using yandex::sdc::control::steering_mpc::RoverState;

    // Minimal stand-in for the production steering model: the checker only reads
    // the wheel base and the (speed-dependent) slip correction from it.
    struct StubSteeringModel {
        double wheel_base = 3.0;
        double slip = 1.0;

        double GetWheelBase() const noexcept { return wheel_base; }
        double SlipCorrection(double /*speed*/) const noexcept { return slip; }
    };

    using Checker = MpcPlanConstraintChecker<StubSteeringModel>;

    // A RoverState carrying only the fields the checker reads: the planar
    // velocity components and the front-wheel steering angle.
    RoverState MakeState(double velocity_x, double fwsa) {
        RoverState state;
        state.velocity_x = velocity_x;
        state.velocity_y = 0.0;
        state.fwsa = fwsa;
        return state;
    }

    struct MpcPlanConstraintCheckerTest: ::testing::Test {
        StubSteeringModel model_{.wheel_base = 3.0, .slip = 1.0};
        // v = 10, fwsa = 0.1, L = 3, slip = 1 -> a_n ~= 3.34 m/s^2.
        // v = 10, fwsa_rate = 0.1                -> jerk_n ~= 3.34 m/s^3.
        Checker::Limits limits_{
            .max_normal_acceleration = 4.0,
            .max_normal_jerk = 10.0,
        };
    };

    TEST_F(MpcPlanConstraintCheckerTest, AdmissibleWhenWithinLimits) {
        Checker checker(model_, limits_);

        const std::vector<RoverState> results{
            MakeState(10.0, 0.09),
            MakeState(10.0, 0.10),
            MakeState(10.0, 0.11),
        };
        const std::vector<double> controls{0.10, 0.10, 0.10}; // fwsa_rate

        EXPECT_TRUE(checker.IsPlanAdmissible(results, controls));
        EXPECT_FALSE(checker.FindViolation(results, controls).has_value());
    }

    TEST_F(MpcPlanConstraintCheckerTest, NormalAccelerationMatchesKinematicFormula) {
        Checker checker(model_, limits_);

        const RoverState state = MakeState(10.0, 0.1);
        const double expected = 10.0 * 10.0 * std::tan(0.1 / 1.0) / 3.0;
        EXPECT_NEAR(checker.NormalAcceleration(state), expected, 1e-9);
    }

    TEST_F(MpcPlanConstraintCheckerTest, NormalJerkMatchesKinematicFormula) {
        Checker checker(model_, limits_);

        const RoverState state = MakeState(10.0, 0.0);
        const double expected = 10.0 * 10.0 * 0.2 / (3.0 * 1.0);
        EXPECT_NEAR(checker.NormalJerk(state, 0.2), expected, 1e-9);
    }

    TEST_F(MpcPlanConstraintCheckerTest, SlipCorrectionScalesTheKinematicAngle) {
        StubSteeringModel model{.wheel_base = 3.0, .slip = 2.0};
        Checker checker(model, limits_);

        // fwsa is divided by slip before tan(): a_n = v^2 * tan(fwsa/slip) / L.
        const RoverState state = MakeState(10.0, 0.2);
        const double expected = 10.0 * 10.0 * std::tan(0.2 / 2.0) / 3.0;
        EXPECT_NEAR(checker.NormalAcceleration(state), expected, 1e-9);
    }

    TEST_F(MpcPlanConstraintCheckerTest, StandingVehicleIsAdmissible) {
        Checker checker(model_, limits_);

        // v = 0 -> both a_n and jerk_n are 0 regardless of steering.
        const std::vector<RoverState> results{MakeState(0.0, 0.5)};
        const std::vector<double> controls{5.0};
        EXPECT_TRUE(checker.IsPlanAdmissible(results, controls));
    }

    TEST_F(MpcPlanConstraintCheckerTest, SkipWhenNormalAccelerationExceeds) {
        Checker checker(model_, limits_);

        // v = 20, fwsa = 0.1 -> a_n ~= 13.4 m/s^2, well above the 4.0 limit.
        const std::vector<RoverState> results{
            MakeState(10.0, 0.10),
            MakeState(20.0, 0.10),
        };
        const std::vector<double> controls{0.10, 0.10};

        EXPECT_FALSE(checker.IsPlanAdmissible(results, controls));

        const auto violation = checker.FindViolation(results, controls);
        ASSERT_TRUE(violation.has_value());
        EXPECT_EQ(violation->kind, Checker::Violation::Kind::NormalAcceleration);
        EXPECT_EQ(violation->index, 1u);
        EXPECT_GT(std::abs(violation->value), violation->limit);
    }

    TEST_F(MpcPlanConstraintCheckerTest, SkipWhenNormalJerkExceeds) {
        Checker checker(model_, limits_);

        // v = 10, fwsa_rate = 0.5 -> jerk_n ~= 16.7 m/s^3, above the 10.0 limit,
        // while the small fwsa keeps a_n within its limit.
        const std::vector<RoverState> results{
            MakeState(10.0, 0.05),
            MakeState(10.0, 0.05),
        };
        const std::vector<double> controls{0.10, 0.50};

        EXPECT_FALSE(checker.IsPlanAdmissible(results, controls));

        const auto violation = checker.FindViolation(results, controls);
        ASSERT_TRUE(violation.has_value());
        EXPECT_EQ(violation->kind, Checker::Violation::Kind::NormalJerk);
        EXPECT_EQ(violation->index, 1u);
    }

    TEST_F(MpcPlanConstraintCheckerTest, EmptyPlanIsAdmissible) {
        Checker checker(model_, limits_);
        EXPECT_TRUE(checker.IsPlanAdmissible({}, {}));
    }

    TEST_F(MpcPlanConstraintCheckerTest, AccelerationIsCheckedBeforeJerk) {
        // A plan that violates both: the acceleration violation comes first in
        // the state order, so it is the one reported.
        Checker checker(model_, limits_);

        const std::vector<RoverState> results{
            MakeState(30.0, 0.2), // huge a_n immediately
            MakeState(10.0, 0.0),
        };
        const std::vector<double> controls{5.0, 5.0}; // also over the jerk limit

        const auto violation = checker.FindViolation(results, controls);
        ASSERT_TRUE(violation.has_value());
        EXPECT_EQ(violation->kind, Checker::Violation::Kind::NormalAcceleration);
        EXPECT_EQ(violation->index, 0u);
    }

} // namespace
