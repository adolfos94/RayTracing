#ifndef CAMERA_H
#define CAMERA_H

#include "vec3.hpp"

class camera
{
public:

	// Camera params
	size_t image_width = 1280; // Rendered image width
	size_t image_height = 720; // Rendered image height

	__device__ camera(size_t width, size_t height) : image_width(width), image_height(height)
	{
		auto focal_length = 1.0;
		auto viewport_height = 2.0;
		auto viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);

		// Calculate the vectors across the horizontal and down the vertical viewport edges.
		auto viewport_u = vec3(viewport_width, 0, 0);
		auto viewport_v = vec3(0, -viewport_height, 0);

		// Calculate the horizontal and vertical delta vectors from pixel to pixel.
		m_pixel_delta_u = viewport_u / image_width;
		m_pixel_delta_v = viewport_v / image_height;

		// Calculate the location of the upper left pixel.
		auto viewport_upper_left = m_camera_center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;

		m_pixel00_loc = viewport_upper_left + 0.5 * (m_pixel_delta_u + m_pixel_delta_v);
	}

	__device__ ray get_ray(size_t i, size_t j) const
	{
		auto pixel_center = m_pixel00_loc + (i * m_pixel_delta_u) + (j * m_pixel_delta_v);
		auto ray_direction = pixel_center - m_camera_center;

		return ray(m_camera_center, ray_direction);
	}

private:
	vec3 m_pixel_delta_u;
	vec3 m_pixel_delta_v;
	vec3 m_pixel00_loc;

	point3 m_camera_center = point3(0, 0, 0);
};

#endif