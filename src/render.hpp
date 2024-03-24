#pragma once

#include "camera.hpp"

namespace render
{
	constexpr size_t SAMPLES_PER_PIXEL = 100;		// Count of random samples for each pixel
	
	inline double linear_to_gamma(double linear_component)
	{
		return sqrt(linear_component);
	}
	
	void sample_color(const size_t samples_per_pixel, color& pixel_color)
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
	
	void render(const hittable& world, const camera& cam, image& img)
	{
		for (size_t j = 0; j < img.height; ++j)
		{
			spdlog::info("Scanlines remaining: {}", (img.height - j));
			
			for (size_t i = 0; i < img.width; ++i)
			{
				color pixel_color(0, 0, 0);
				
				for (size_t s = 0; s < SAMPLES_PER_PIXEL; ++s)
				{
					ray ray = cam.get_ray(i, j);
					pixel_color += cam.ray_color(ray, world, 0);
				}
				
				sample_color(SAMPLES_PER_PIXEL, pixel_color);
				
				img.data[(j * img.width + i) * 3 + 0] = static_cast<uint8_t>(pixel_color.x());
				img.data[(j * img.width + i) * 3 + 1] = static_cast<uint8_t>(pixel_color.y());
				img.data[(j * img.width + i) * 3 + 2] = static_cast<uint8_t>(pixel_color.z());
			}
		}
	}
	
} // namespace render
