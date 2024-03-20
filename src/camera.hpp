#ifndef CAMERA_H
#define CAMERA_H

#include "color.hpp"
#include "hittable.hpp"

class camera
{
public:

	// Camera params
	double vfov = 90;					// Vertical Field of View
	point3 lookfrom = point3(0, 0, 0); // Point camera is looking from
	point3 lookat = point3(0, 0, -1);	// Point camera is looking at
	vec3 up_vector = vec3(0, 1, 0);		// Camera "up" vector (direction)

	camera(const size_t width, const size_t height) : m_width(width), m_height(height){};

	void render(const hittable& world, const rerun::RecordingStream& rec)
	{
		initialize();

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

	size_t m_width;           // Rendered image width
	size_t m_height;          // Rendered image height

	size_t m_samples_per_pixel = 100; // Count of random samples for each pixel
	size_t m_max_depth_ray_generation = 50; // Maximum number of ray bounces into scene

	point3 m_camera_center; // Camera center
	point3 m_pixel_init_00; // Location of pixel 0, 0

	vec3 m_pixel_delta_u; // Offset to pixel to the right
	vec3 m_pixel_delta_v; // Offset to pixel below
	vec3 m_u, m_v, m_w;	  // Camera frame basis vectors

	// Visualizations
	Image m_image;
	Rays3D m_scene_rays;

	void initialize()
	{
		m_image = Image(m_width, m_height);

		m_camera_center = lookfrom;

		// Determine viewport dimensions.
		auto focal_length = (lookfrom - lookat).length();
		auto tetha = degrees_to_radians(vfov);
		auto h = tan(tetha / 2.0);
		auto viewport_height = 2 * h * focal_length;
		auto viewport_width = viewport_height * (static_cast<double>(m_width) / m_height);

		// Calculate the u,v,w unit basis vectors for the camera coordinate frame.
		m_w = unit_vector(lookfrom - lookat);
		m_u = unit_vector(cross(up_vector, m_w));
		m_v = cross(m_w, m_u);

		// Calculate the vectors across the horizontal and down the vertical viewport edges.
		vec3 viewport_u = viewport_width * m_u;    // Vector across viewport horizontal edge
		vec3 viewport_v = viewport_height * -m_v;  // Vector down viewport vertical edge

		// Calculate the horizontal and vertical delta vectors from pixel to pixel.
		m_pixel_delta_u = viewport_u / m_width;
		m_pixel_delta_v = viewport_v / m_height;

		// Calculate the location of the upper left pixel.
		auto viewport_upper_left = m_camera_center - (focal_length * m_w) - viewport_u / 2 - viewport_v / 2;
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

	color ray_color(const ray& r, const hittable& world, size_t depth) const
	{
		hit_record rec;

		if (depth >= m_max_depth_ray_generation)
			return color();

		if (world.hit(r, interval(0.001, infinity), rec))
		{
			ray scattered;
			color attenuation;

			if (rec.mat->scatter(r, rec, attenuation, scattered))
				return attenuation * ray_color(scattered, world, depth + 1);

			return color();
		}

		vec3 unit_direction = unit_vector(r.direction());
		auto a = 0.5 * (unit_direction.y() + 1.0);

		return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
	}

	inline double linear_to_gamma(double linear_component)
	{
		return sqrt(linear_component);
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

		// Apply the linear to gamma transform.
		r = linear_to_gamma(r);
		g = linear_to_gamma(g);
		b = linear_to_gamma(b);

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
			spdlog::info("Scanlines remaining: {}", (m_image.height - j));

			for (size_t i = 0; i < m_image.width; ++i)
			{
				color pixel_color(0, 0, 0);

				for (size_t s = 0; s < m_samples_per_pixel; ++s)
				{
					ray ray = get_ray(i, j);
					pixel_color += ray_color(ray, world, 0);
				}

				m_image.draw(i, j, sample_color(pixel_color));
			}
		}
	}
};

#endif