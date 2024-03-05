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


// Structs
struct Image
{
	size_t width;
	size_t height;
	std::vector<uint8_t> data;

	Image() {};

	Image(const size_t w, const size_t h, uint8_t val = 0) : width(w), height(h)
	{
		data.resize(width * height * 3, val);
	}

	void draw(const size_t i, const size_t j, const rerun::Color color)
	{
		data[(j * width + i) * 3 + 0] = color.r();
		data[(j * width + i) * 3 + 1] = color.g();
		data[(j * width + i) * 3 + 2] = color.b();
	}
};

struct Rays3D
{
	std::vector<rerun::Position3D> origins;
	std::vector<rerun::Vector3D> vectors;
	std::vector<rerun::Color> colors;
};