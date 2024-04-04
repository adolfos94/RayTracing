#ifndef COLOR_H
#define COLOR_H

#include "ray.hpp"
#include "hittable_list.hpp"

using color = vec3;

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

	// Write the translated [0,255] value of each color component.
	const interval intensity(0.000, 0.999);
	pixel_color[0] = 256.0 * intensity.clamp(r);
	pixel_color[1] = 256.0 * intensity.clamp(g);
	pixel_color[2] = 256.0 * intensity.clamp(b);
}

__device__ inline color ray_color(const ray& r, hittable_list** world)
{
	hit_record rec;
	if ((*world)->hit(r, interval(0, infinity), rec))
		return 0.5 * (rec.normal + color(1, 1, 1));

	vec3 unit_direction = unit_vector(r.direction());
	auto a = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}

#endif