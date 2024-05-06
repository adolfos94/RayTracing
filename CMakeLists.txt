cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)

project(RayTracing VERSION "1.0.0" LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(spdlog REQUIRED)

# Rerun
set(RERUN_ARROW_LINK_SHARED ON)
set(RERUN_DOWNLOAD_AND_BUILD_ARROW OFF)

include(FetchContent)
FetchContent_Declare(rerun_sdk URL
    https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)

add_subdirectory(src)