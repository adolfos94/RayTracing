#ifndef COLOR_H
#define COLOR_H

#include "vec3.hpp"
#include "interval.hpp"

using color = vec3;

/// <summary>
/// Samples color based on samples per pixel
/// </summary>
/// <param name="samples_per_pixel">Number of samples per pixel</param>
/// <param name="pixel_color">[0, 255] value of each color component</param>
inline void sample_color(const size_t samples_per_pixel, color& pixel_color)
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

	// Write the translated [0, 255] value of each color component
	pixel_color[0] = 256.0 * intensity.clamp(r);
	pixel_color[1] = 256.0 * intensity.clamp(g);
	pixel_color[2] = 256.0 * intensity.clamp(b);
}

#endif