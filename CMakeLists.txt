cmake_minimum_required(VERSION 3.22)
project(cuTrace LANGUAGES CXX CUDA VERSION 1.1)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(CMAKE_CXX_STANDARD 17)
set(CC "75" CACHE STRING "Which compute capability to build for? E.g. 61, 70, 75, 86")

include_directories(.)

add_executable(cuTrace
        cuTrace.cpp
        cuTrace.cu
        cuTrace.cuh)
