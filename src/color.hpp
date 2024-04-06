#ifndef COLOR_H
#define COLOR_H

#include "ray.hpp"
#include "hittable_list.hpp"

using color = vec3;

__device__ inline double linear_to_gamma(double linear_component)
{
	return sqrt(linear_component);
}

__device__ inline void sample_color(const size_t samples_per_pixel, color& pixel_color)
{
	auto r = pixel_color.x();
	auto g = pixel_color.y();
	auto b = pixel_color.z();

	// Divide the color by the number of samples.
	auto scale = 1.0 / samples_per_pixel;
	r *= scale;
	g *= scale;
	b *= scale;

	// Apply the linear to gamma transform.
	r = linear_to_gamma(r);
	g = linear_to_gamma(g);
	b = linear_to_gamma(b);

	// Write the translated [0,255] value of each color component.
	const interval intensity(0.000, 0.999);
	pixel_color[0] = 256.0 * intensity.clamp(r);
	pixel_color[1] = 256.0 * intensity.clamp(g);
	pixel_color[2] = 256.0 * intensity.clamp(b);
}

#endif