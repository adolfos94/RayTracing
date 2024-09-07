cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(RayTracing VERSION "1.0.0" LANGUAGES CXX CUDA)

add_subdirectory(src)