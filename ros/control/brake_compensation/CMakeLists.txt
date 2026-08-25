# Standalone CMake project for building and running the unit tests for
# BrakeCompensationBuilder in isolation. The production monorepo this
# component lives in most likely builds it with its own build system
# (ya.make/Bazel/whatever "ros/control/common" and "core/math" belong to);
# this file exists so the tests can be built and run here, without the rest
# of that monorepo checked out.
#
# tests/stub_include holds minimal, test-only stand-ins for the two external
# headers acceleration_matrix.h depends on (ros/control/common/struct_interpolant.h,
# core/math/butterworth_filter.h), which are not part of this checkout. They
# are not the production implementation -- see the comments in those files.
cmake_minimum_required(VERSION 3.16)
project(brake_compensation CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(acceleration_matrix STATIC acceleration_matrix.cpp)
target_include_directories(acceleration_matrix PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/stub_include)

if(BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
    find_package(GTest REQUIRED)
    enable_testing()

    add_executable(acceleration_matrix_test tests/acceleration_matrix_test.cpp)
    target_link_libraries(acceleration_matrix_test PRIVATE
        acceleration_matrix
        GTest::gtest
        GTest::gtest_main)

    add_test(NAME acceleration_matrix_test COMMAND acceleration_matrix_test)
endif()
