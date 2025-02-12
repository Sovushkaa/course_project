cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic")

find_package(Eigen3 3.3 REQUIRED)

if(Eigen3_FOUND)
    message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Eigen3 not found!")
endif()

add_executable(network main.cpp network.cpp layer.cpp penalty_function.cpp)

target_link_libraries(network Eigen3::Eigen)

include_directories(${EIGEN3_INCLUDE_DIR})