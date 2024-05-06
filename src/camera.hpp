#ifndef CAMERA_H
#define CAMERA_H

#include "vec3.hpp"
#include "ray.hpp"

constexpr size_t SAMPLES_PER_PIXEL = 500; // Count of random samples for each pixel
constexpr size_t MAX_DEPTH = 50; // Maximum number of ray bounces into scene

class camera
{
public:

	// Camera params
	size_t image_width = 1280; // Rendered image width
	size_t image_height = 720; // Rendered image height
	double vfov = 90; // Vertical view angle (field of view)
	point3 lookfrom = point3(0, 0, -1); // Point camera is looking from
	point3 lookat = point3(0, 0, 0); // Point camera is looking at
	const vec3 vup = vec3(0, 1, 0); // Camera-relative "up" direction

	double defocus_angle = 0; // Variation angle of rays through each pixel
	double focus_dist = 10; // Distance from camera lookfrom point to plane of perfect focus

	__device__ camera(size_t width, size_t height) : image_width(width), image_height(height) {}

	__device__ void initialize()
	{
		m_camera_center = lookfrom;

		// Determine viewport dimensions.
		auto theta = degrees_to_radians(vfov);
		auto h = tan(theta / 2);
		auto viewport_height = 2 * h * focus_dist;
		auto viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);

		// Calculate the u,v,w unit basis vectors for the camera coordinate frame.
		m_w = unit_vector(lookfrom - lookat);
		m_u = unit_vector(cross(vup, m_w));
		m_v = cross(m_w, m_u);

		// Calculate the vectors across the horizontal and down the vertical viewport edges.
		vec3 viewport_u = viewport_width * m_u;    // Vector across viewport horizontal edge
		vec3 viewport_v = viewport_height * -m_v;  // Vector down viewport vertical edge

		// Calculate the horizontal and vertical delta vectors from pixel to pixel.
		m_pixel_delta_u = viewport_u / image_width;
		m_pixel_delta_v = viewport_v / image_height;

		// Calculate the location of the upper left pixel.
		auto viewport_upper_left = m_camera_center - (focus_dist * m_w) - viewport_u / 2 - viewport_v / 2;
		m_pixel00_loc = viewport_upper_left + 0.5 * (m_pixel_delta_u + m_pixel_delta_v);

		// Calculate the camera defocus disk basis vectors.
		auto defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
		m_defocus_disk_u = m_u * defocus_radius;
		m_defocus_disk_v = m_v * defocus_radius;
	}

	// Construct a camera ray originating from the defocus disk and directed at a randomly
	// sampled point around the pixel location i, j.
	__device__ ray get_ray(size_t i, size_t j, curandState* local_rand_state) const
	{
		auto pixel_center = m_pixel00_loc + (i * m_pixel_delta_u) + (j * m_pixel_delta_v);
		auto pixel_sample = pixel_center + pixel_sample_square(local_rand_state);

		auto ray_origin = (defocus_angle <= 0) ? m_camera_center : defocus_disk_sample(local_rand_state);
		auto ray_direction = pixel_sample - ray_origin;
		auto ray_time = random_double(local_rand_state);

		return ray(ray_origin, ray_direction, ray_time);
	}

	__device__ color ray_color(const ray& r, hittable_list** world, curandState* local_rand_state) const
	{
		ray current_ray = r;
		vec3 current_attenuation = vec3(1.0, 1.0, 1.0);

		for (size_t i = 0; i < MAX_DEPTH; ++i)
		{
			hit_record rec;
			if (!(*world)->hit(current_ray, interval(0.001, infinity), rec))
				break;

			ray scattered;
			vec3 attenuation;

			if (!rec.mat->scatter(current_ray, rec, attenuation, scattered, local_rand_state))
				break;

			current_ray = scattered;
			current_attenuation *= attenuation;
		}

		// If we've exceeded the ray bounce limit, no more light is gathered.
		vec3 unit_direction = unit_vector(current_ray.direction());
		auto a = 0.5 * (unit_direction.y() + 1.0);
		auto c = (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
		return current_attenuation * c;
	}

private:
	point3 m_camera_center; // Camera center
	vec3 m_pixel_delta_u; // Offset to pixel to the right
	vec3 m_pixel_delta_v; // Offset to pixel below
	vec3 m_pixel00_loc; // Location of pixel 0, 0
	vec3 m_u, m_v, m_w; // Camera frame basis vectors

	vec3 m_defocus_disk_u;  // Defocus disk horizontal radius
	vec3 m_defocus_disk_v;  // Defocus disk vertical radius

	__device__ vec3 pixel_sample_square(curandState* local_rand_state) const
	{
		// Returns a random point in the square surrounding a pixel at the origin.
		auto px = -0.5 + random_double(local_rand_state);
		auto py = -0.5 + random_double(local_rand_state);

		return (px * m_pixel_delta_u) + (py * m_pixel_delta_v);
	}

	__device__ point3 defocus_disk_sample(curandState* local_rand_state) const
	{
		// Returns a random point in the camera defocus disk.
		auto p = random_in_unit_disk(local_rand_state);
		return m_camera_center + (p[0] * m_defocus_disk_u) + (p[1] * m_defocus_disk_v);
	}
};

#endif