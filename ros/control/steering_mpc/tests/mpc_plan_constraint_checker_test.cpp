#include <gtest/gtest.h>

#include "sdg/sdc/ros/control/steering_mpc/mpc_plan_constraint_checker.h"

#include <cmath>
#include <vector>

namespace {
    using yandex::sdc::control::steering_mpc::MpcPlanConstraintChecker;
    using yandex::sdc::control::steering_mpc::RoverState;

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
        // v^2 * tan(fwsa) / L with v = 10, fwsa = 0.1, L = 3 -> ~3.34 m/s^2.
        MpcPlanConstraintChecker::Limits limits_{
            .max_normal_acceleration = 4.0,
            .max_normal_jerk = 10.0,
            .wheelbase = 3.0,
            .time_step = 0.1,
        };
    };

    TEST_F(MpcPlanConstraintCheckerTest, AdmissibleWhenWithinLimits) {
        MpcPlanConstraintChecker checker(limits_);

        const std::vector<RoverState> results{
            MakeState(10.0, 0.09),
            MakeState(10.0, 0.10),
            MakeState(10.0, 0.11),
        };
        const std::vector<double> controls{0.09, 0.10, 0.11};

        EXPECT_TRUE(checker.IsPlanAdmissible(results, controls));
        EXPECT_FALSE(checker.FindViolation(results, controls).has_value());
    }

    TEST_F(MpcPlanConstraintCheckerTest, NormalAccelerationMatchesKinematicFormula) {
        MpcPlanConstraintChecker checker(limits_);

        const RoverState state = MakeState(10.0, 0.1);
        const double expected = 10.0 * 10.0 * std::tan(0.1) / 3.0;
        EXPECT_NEAR(checker.NormalAcceleration(state), expected, 1e-9);
    }

    TEST_F(MpcPlanConstraintCheckerTest, SkipWhenNormalAccelerationExceeds) {
        MpcPlanConstraintChecker checker(limits_);

        // v = 20, fwsa = 0.1 -> a_n ~= 13.4 m/s^2, well above the 4.0 limit.
        const std::vector<RoverState> results{
            MakeState(10.0, 0.10),
            MakeState(20.0, 0.10),
        };
        const std::vector<double> controls{0.10, 0.10};

        EXPECT_FALSE(checker.IsPlanAdmissible(results, controls));

        const auto violation = checker.FindViolation(results, controls);
        ASSERT_TRUE(violation.has_value());
        EXPECT_EQ(violation->kind,
                  MpcPlanConstraintChecker::Violation::Kind::NormalAcceleration);
        EXPECT_EQ(violation->index, 1u);
        EXPECT_GT(std::abs(violation->value), violation->limit);
    }

    TEST_F(MpcPlanConstraintCheckerTest, SkipWhenNormalJerkExceeds) {
        // Both states are within the acceleration limit, but a_n jumps between
        // them fast enough that its derivative (normal jerk) blows the limit.
        // a_n(0.0) = 0, a_n(0.1) ~= 3.34 over dt = 0.1 -> jerk ~= 33 m/s^3.
        MpcPlanConstraintChecker checker(limits_);

        const std::vector<RoverState> results{
            MakeState(10.0, 0.0),
            MakeState(10.0, 0.1),
        };
        const std::vector<double> controls{0.0, 0.1};

        EXPECT_FALSE(checker.IsPlanAdmissible(results, controls));

        const auto violation = checker.FindViolation(results, controls);
        ASSERT_TRUE(violation.has_value());
        EXPECT_EQ(violation->kind,
                  MpcPlanConstraintChecker::Violation::Kind::NormalJerk);
        EXPECT_EQ(violation->index, 0u);
    }

    TEST_F(MpcPlanConstraintCheckerTest, EmptyPlanIsAdmissible) {
        MpcPlanConstraintChecker checker(limits_);
        EXPECT_TRUE(checker.IsPlanAdmissible({}, {}));
    }

    TEST_F(MpcPlanConstraintCheckerTest, SingleStatePlanHasNoJerk) {
        MpcPlanConstraintChecker checker(limits_);

        const std::vector<RoverState> results{MakeState(10.0, 0.1)};
        EXPECT_TRUE(checker.IsPlanAdmissible(results, {0.1}));
    }

    TEST_F(MpcPlanConstraintCheckerTest, AccelerationIsCheckedBeforeJerk) {
        // A plan that violates both: the acceleration violation comes first in
        // the state order, so it is the one reported.
        MpcPlanConstraintChecker checker(limits_);

        const std::vector<RoverState> results{
            MakeState(30.0, 0.2), // huge a_n immediately
            MakeState(10.0, 0.0),
        };

        const auto violation = checker.FindViolation(results, {0.2, 0.0});
        ASSERT_TRUE(violation.has_value());
        EXPECT_EQ(violation->kind,
                  MpcPlanConstraintChecker::Violation::Kind::NormalAcceleration);
        EXPECT_EQ(violation->index, 0u);
    }

} // namespace
