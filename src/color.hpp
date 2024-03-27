#ifndef COLOR_H
#define COLOR_H

#include "vec3.hpp"
#include "ray.hpp"

using color = vec3;

__device__ inline void sample_color(color& pixel_color)
{
	// Write the translated [0,255] value of each color component.
	pixel_color[0] = 255.999 * pixel_color.x();
	pixel_color[1] = 255.999 * pixel_color.y();
	pixel_color[2] = 255.999 * pixel_color.z();
}

__device__ inline color ray_color(const ray& r)
{
	vec3 unit_direction = unit_vector(r.direction());
	auto a = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}

#endif