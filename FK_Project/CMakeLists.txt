cmake_minimum_required(VERSION 3.12.0)
project(FK-Project LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(Fisher-Kolmogorov-Solver src/main.cpp src/Fisher_Kolmogorov.cpp)
deal_ii_setup_target(Fisher-Kolmogorov-Solver)
