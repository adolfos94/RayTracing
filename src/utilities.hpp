#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include <rerun.hpp>
#include <spdlog/spdlog.h>

// Constants
__constant__ const double infinity = std::numeric_limits<double>::infinity();
__constant__ const double pi = 3.1415926535897932385;

// Utility Functions

__device__ inline double degrees_to_radians(double degrees)
{
	return degrees * pi / 180.0;
}

__device__ inline double random_double(curandState& local_rand_state)
{
	// Returns a random real in [0,1).
	return curand_uniform(&local_rand_state);
}

__device__ inline double random_double(curandState& local_rand_state, double min, double max)
{
	// Returns a random real in [min,max).
	return min + (max - min) * random_double(local_rand_state);
}

// Structs
struct image
{
	size_t width = 0;
	size_t height = 0;
	size_t size = 0;
	uint8_t* data = nullptr;

	image() {};

	image(const size_t w, const size_t h, uint8_t val = 0) : width(w), height(h)
	{
		size = width * height * 3;
		data = new uint8_t[size]();
	}
};