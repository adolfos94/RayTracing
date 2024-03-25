#pragma once

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <rerun.hpp>
#include <spdlog/spdlog.h>

// Constants
constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double pi = 3.1415926535897932385;

// Utility Functions

/// <summary>
/// Converts degrees into radians
/// </summary>
/// <param name="degrees"></param>
/// <returns></returns>
inline double degrees_to_radians(double degrees)
{
	return degrees * pi / 180.0;
}

/// <summary>
/// Returns a random real number in [0.0, 1.0).
/// </summary>
/// <returns></returns>
inline double random_double()
{
	static std::default_random_engine generator;
	static std::uniform_real_distribution<double> distribution(0.0, 1.0);

	return distribution(generator);
}

/// <summary>
/// Returns a random real number in [min,max).
/// </summary>
/// <param name="min"></param>
/// <param name="max"></param>
/// <returns></returns>
inline double random_double(double min, double max)
{
	return min + (max - min) * random_double();
}

/// <summary>
/// Linear to gamma transform
/// </summary>
/// <param name="linear_component"></param>
/// <returns></returns>
inline double linear_to_gamma(double linear_component)
{
	return sqrt(linear_component);
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