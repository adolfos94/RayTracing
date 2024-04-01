#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_vector.h>

#include <rerun.hpp>
#include <spdlog/spdlog.h>

// Constants

__constant__ const double infinity = std::numeric_limits<double>::infinity();
__constant__ const double pi = 3.1415926535897932385;

// Utility Functions

inline double degrees_to_radians(double degrees)
{
	return degrees * pi / 180.0;
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