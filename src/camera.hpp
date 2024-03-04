#ifndef CAMERA_H
#define CAMERA_H

#include "color.hpp"
#include "hittable.hpp"

class camera
{
public:

	camera(const size_t width, const size_t height) :
		m_width(width), m_height(height)
	{
		initialize();
	};

	void render(const hittable& world, const rerun::RecordingStream& rec)
	{
		ray_tracing_scene(world);

		//world.render(rec);

		rec.log_timeless(
			"world/image",
			rerun::Image({ m_image.height, m_image.width, 3 }, m_image.data));

		/*rec.log_timeless(
			"world/rays",
			rerun::Arrows3D::from_vectors(m_scene_rays.vectors).
			with_origins(m_scene_rays.origins).
			with_colors(m_scene_rays.colors)
		);*/
	}

private:

	// Camera params
	size_t m_width;           // Rendered image width
	size_t m_height;          // Rendered image height
	double m_focal_length;    // Camera focal length
	double m_viewport_width;  // Camera view port width
	double m_viewport_height; // Camera view port height

	size_t m_samples_per_pixel = 100; // Count of random samples for each pixel

	point3 m_camera_center = point3(0.0, 0.0, 0.0); // Camera center

	vec3 m_pixel_delta_u; // Offset to pixel to the right
	vec3 m_pixel_delta_v; // Offset to pixel below
	vec3 m_pixel_init_00; // Location of pixel 0, 0

	// Visualizations
	Image m_image;
	Rays3D m_scene_rays;

	void initialize()
	{
		m_image = Image(m_width, m_height);

		// Determine viewport dimensions.
		m_focal_length = 1.0;
		m_viewport_height = 2.0;
		m_viewport_width = m_viewport_height * (static_cast<double>(m_width) / m_height);

		// Calculate the vectors across the horizontal and down the vertical viewport edges.
		auto viewport_u = vec3(m_viewport_width, 0, 0);
		auto viewport_v = vec3(0, -m_viewport_height, 0);

		// Calculate the horizontal and vertical delta vectors from pixel to pixel.
		m_pixel_delta_u = viewport_u / m_width;
		m_pixel_delta_v = viewport_v / m_height;

		// Calculate the location of the upper left pixel.
		auto viewport_upper_left = m_camera_center - vec3(0, 0, m_focal_length) - viewport_u / 2 - viewport_v / 2;
		m_pixel_init_00 = viewport_upper_left + 0.5 * (m_pixel_delta_u + m_pixel_delta_v);
	}

	/// <summary>
	/// Returns a random point in the square surrounding a pixel at the origin.
	/// </summary>
	vec3 pixel_sample_square() const 
	{
		auto px = -0.5 + random_double();
		auto py = -0.5 + random_double();
		return (px * m_pixel_delta_u) + (py * m_pixel_delta_v);
	}

	/// <summary>
	/// Get a randomly sampled camera ray for the pixel at location i, j.
	/// </summary>
	ray get_ray(size_t i, size_t j) const
	{
		auto pixel_center = m_pixel_init_00 + (i * m_pixel_delta_u) + (j * m_pixel_delta_v);
		auto pixel_sample = pixel_center + pixel_sample_square();

		auto ray_origin = m_camera_center;
		auto ray_direction = pixel_sample - ray_origin;

		return ray(ray_origin, ray_direction);
	}

	color ray_color(const ray& r, const hittable& world) const
	{
		hit_record rec;

		if (world.hit(r, interval(0, infinity), rec))
			return 0.5 * (rec.normal + color(1, 1, 1));

		vec3 unit_direction = unit_vector(r.direction());
		auto a = 0.5 * (unit_direction.y() + 1.0);
		return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
	}


	rerun::Color sample_color(color pixel_color) 
	{
		auto r = pixel_color.x();
		auto g = pixel_color.y();
		auto b = pixel_color.z();

		// Divide the color by the number of samples.
		auto scale = 1.0 / m_samples_per_pixel;
		r *= scale;
		g *= scale;
		b *= scale;

		// Write the translated [0,255] value of each color component
		return rerun::Color(
			static_cast<uint8_t>(256 * intensity.clamp(r)),
			static_cast<uint8_t>(256 * intensity.clamp(g)),
			static_cast<uint8_t>(256 * intensity.clamp(b))
		);
	}


	void ray_tracing_scene(const hittable& world)
	{
		for (size_t j = 0; j < m_image.height; ++j)
		{
			for (size_t i = 0; i < m_image.width; ++i)
			{
				color pixel_color(0, 0, 0);

				for (size_t s = 0; s < m_samples_per_pixel; ++s)
				{
					ray ray = get_ray(i, j);
					pixel_color += ray_color(ray, world);
				}

				m_image.draw(i, j, sample_color(pixel_color));
			}
		}
	}
};

#endif