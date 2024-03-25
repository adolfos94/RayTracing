#ifndef CAMERA_H
#define CAMERA_H

#include "color.hpp"
#include "hittable.hpp"

constexpr size_t SAMPLES_PER_PIXEL = 100;		// Count of random samples for each pixel

class camera
{
public:
	
	// Camera params
	double vfov = 90;				   		// Vertical Field of View
	double defocus_angle = 0;		   		// Variation angle of rays through each pixel
	double focus_distance = 10;		   		// Distance from camera loookfrom point to plane
	size_t width = 1280;       		   		// Rendered image width
	size_t height = 720;               		// Rendered image height
	point3 lookfrom = point3(0, 0, 0); 		// Point camera is looking from
	point3 lookat = point3(0, 0, -1);  		// Point camera is looking at
	const vec3 up_vector = vec3(0, 1, 0);	// Camera "up" vector (direction)
	
	camera() {};
	
	/// <summary>
	/// Initializes viewport dimensions and camera coordinate frame.
	/// </summary>
	void initialize()
	{
		m_camera_center = lookfrom;
		
		// Determine viewport dimensions.
		auto tetha = degrees_to_radians(vfov);
		auto h = tan(tetha / 2.0);
		auto viewport_height = 2 * h * focus_distance;
		auto viewport_width = viewport_height * (static_cast<double>(width) / height);
		
		// Calculate the u,v,w unit basis vectors for the camera coordinate frame.
		m_w = unit_vector(lookfrom - lookat);
		m_u = unit_vector(cross(up_vector, m_w));
		m_v = cross(m_w, m_u);
		
		// Calculate the vectors across the horizontal and down the vertical viewport edges.
		vec3 viewport_u = viewport_width * m_u;    // Vector across viewport horizontal edge
		vec3 viewport_v = viewport_height * -m_v;  // Vector down viewport vertical edge
		
		// Calculate the horizontal and vertical delta vectors from pixel to pixel.
		m_pixel_delta_u = viewport_u / width;
		m_pixel_delta_v = viewport_v / height;
		
		// Calculate the location of the upper left pixel.
		auto viewport_upper_left = m_camera_center - (focus_distance * m_w) - viewport_u / 2 - viewport_v / 2;
		m_pixel_init_00 = viewport_upper_left + 0.5 * (m_pixel_delta_u + m_pixel_delta_v);
		
		// Calculate the camera defocus disk basis vectors.
		auto defocus_radius = focus_distance * std::tan(degrees_to_radians(defocus_angle / 2));
		m_defocus_disk_u = m_u * defocus_radius;
		m_defocus_disk_v = m_v * defocus_radius;
	}
	
	/// <summary>
	/// Get a randomly sampled camera ray for the pixel at location i, j.
	/// </summary>
	ray get_ray(size_t i, size_t j) const
	{
		auto pixel_center = m_pixel_init_00 + (i * m_pixel_delta_u) + (j * m_pixel_delta_v);
		auto pixel_sample = pixel_center + pixel_sample_square();
		
		auto ray_origin = (defocus_angle <= 0) ? m_camera_center : defocus_disk_sample();;
		auto ray_direction = pixel_sample - ray_origin;
		
		return ray(ray_origin, ray_direction);
	}
	
	/// <summary>
	/// Returns the color corresponding to the ray
	/// </summary>
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
	
private:
	
	size_t m_max_depth_ray_generation = 50; // Maximum number of ray bounces into scene
	
	point3 m_camera_center; // Camera center
	point3 m_pixel_init_00; // Location of pixel 0, 0
	
	vec3 m_pixel_delta_u; // Offset to pixel to the right
	vec3 m_pixel_delta_v; // Offset to pixel below
	vec3 m_u, m_v, m_w;	  // Camera frame basis vectors
	
	vec3 m_defocus_disk_u; // Defocus disk horizontal radius
	vec3 m_defocus_disk_v; // Defocus disk vertical radius
	
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
	/// Returns a random point in the camera defocus disk.
	/// </summary>
	point3 defocus_disk_sample() const
	{
		auto p = random_in_unit_disk();
		return m_camera_center + (p[0] * m_defocus_disk_u) + (p[1] * m_defocus_disk_v);
	}
};

#endif
