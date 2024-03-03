#pragma once

#include <cmath>
#include <limits>
#include <vector>

#include <rerun.hpp>

#include "color.hpp"
#include "ray.hpp"
#include "vec3.hpp"

// Constants
constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double pi = 3.1415926535897932385;

// Utility Functions
inline double degrees_to_radians(double degrees)
{
	return degrees * pi / 180.0;
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

	void draw(const size_t i, const size_t j, const color pixel_color)
	{
		data[(i * width + j) * 3 + 0] = r(pixel_color);
		data[(i * width + j) * 3 + 1] = g(pixel_color);
		data[(i * width + j) * 3 + 2] = b(pixel_color);
	}
};

struct Rays3D
{
	std::vector<rerun::Position3D> origins;
	std::vector<rerun::Vector3D> vectors;
	std::vector<rerun::Color> colors;
};