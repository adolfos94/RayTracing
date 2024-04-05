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

__device__ inline color ray_color(const ray& r, const hittable_list** world, curandState* local_rand_state)
{
	ray current_ray = r;
	vec3 current_attenuation = vec3(1.0, 1.0, 1.0);

	for (size_t i = 0; i < MAX_DEPTH; ++i)
	{
		hit_record rec;
		if ((*world)->hit(current_ray, interval(0.001, infinity), rec))
		{
			ray scattered;
			vec3 attenuation;
			if (rec.mat->scatter(current_ray, rec, attenuation, scattered, local_rand_state))
			{
				current_ray = scattered;
				current_attenuation *= attenuation;
			}
			else return color();
		}
		else
		{
			vec3 unit_direction = unit_vector(current_ray.direction());
			auto a = 0.5 * (unit_direction.y() + 1.0);
			auto c = (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
			return current_attenuation * c;
		}
	}

	// If we've exceeded the ray bounce limit, no more light is gathered.
	return color();
}

#endif