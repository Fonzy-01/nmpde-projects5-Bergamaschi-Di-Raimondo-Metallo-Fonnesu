cmake_minimum_required(VERSION 3.12.0)
project(Fisker_Kolmogorov_Solver LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(FK_Project src/main.cpp src/Fisher_Kolmogorov.cpp src/Fisher1D.cpp)
deal_ii_setup_target(FK_Project)
