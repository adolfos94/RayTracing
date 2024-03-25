#pragma once

#include "camera.hpp"

namespace render
{
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