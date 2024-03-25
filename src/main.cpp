#include "camera.hpp"
#include "hittables.hpp"
#include "utilities.hpp"

#include "geometries/sphere.hpp"
#include "materials/dielectric.hpp"
#include "materials/lambertian.hpp"
#include "materials/metal.hpp"

#include "render.hpp"
#include "cuda/render.hpp"

hittable_list RayTracingInOneWeekendWorld()
{
	hittable_list world;

	auto ground_material = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
	world.add(std::make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

	for (int a = -11; a < 11; a++)
	{
		for (int b = -11; b < 11; b++)
		{
			auto choose_mat = random_double();
			point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

			if ((center - point3(4, 0.2, 0)).length() > 0.9)
			{
				std::shared_ptr<material> sphere_material;

				if (choose_mat < 0.8) {
					// diffuse
					auto albedo = color::random() * color::random();
					sphere_material = std::make_shared<lambertian>(albedo);
					world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
				}
				else if (choose_mat < 0.95) {
					// metal
					auto albedo = color::random(0.5, 1);
					auto fuzz = random_double(0, 0.5);
					sphere_material = std::make_shared<metal>(albedo, fuzz);
					world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
				}
				else {
					// glass
					sphere_material = std::make_shared<dielectric>(1.5);
					world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
				}
			}
		}
	}

	auto material1 = std::make_shared<dielectric>(1.5);
	world.add(std::make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

	auto material2 = std::make_shared<lambertian>(color(0.4, 0.2, 0.1));
	world.add(std::make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

	auto material3 = std::make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
	world.add(std::make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

	return world;
}

int main()
{
	// ReRun visualization
	const auto rec = rerun::RecordingStream("RayTracing");
	rec.connect().exit_on_failure();

	// **** Camera ****
	camera cam = camera();
	cam.width = 1280;
	cam.height = 720;
	cam.vfov = 20;
	//	cam.lookfrom = point3(13, 2, 3);
	//	cam.lookat = point3(0, 0, 0);
	cam.lookfrom = point3(-2, 2, 1);
	cam.lookat = point3(0, 0, -1);

	//	cam.defocus_angle = 0.6;
	//	cam.focus_distance = 10.0;
	cam.defocus_angle = 10.0;
	cam.focus_distance = 3.4;
	cam.initialize();

	hittable_list world;

	auto material_ground = std::make_shared<lambertian>(color(0.8, 0.8, 0.0));
	auto material_center = std::make_shared<lambertian>(color(0.1, 0.2, 0.5));
	auto material_left = std::make_shared<dielectric>(1.5);
	auto material_right = std::make_shared<metal>(color(0.8, 0.6, 0.2), 0.0);

	world.add(std::make_shared<sphere>(point3(0.0, -100.5, -1.0), 100.0, material_ground));
	world.add(std::make_shared<sphere>(point3(0.0, 0.0, -1.0), 0.5, material_center));
	world.add(std::make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.5, material_left));
	world.add(std::make_shared<sphere>(point3(-1.0, 0.0, -1.0), -0.4, material_left));
	world.add(std::make_shared<sphere>(point3(1.0, 0.0, -1.0), 0.5, material_right));

	image img = image(cam.width, cam.height);

	// Render scene
	render::render(world, cam, img);
	rec.log_timeless("world/image", rerun::Image({ img.height, img.width, 3 }, img.data));

	// Render scene with CUDA

	image d_img;
	d_img.width = img.width;
	d_img.height = img.height;
	cudaMalloc(&d_img.data, img.size);

	render::cuda::get_device_params();
	render::cuda::render(d_img);

	cudaMemcpy(img.data, d_img.data, img.size, cudaMemcpyDeviceToHost);
	cudaFree(d_img.data);

	rec.log_timeless("world/cuda_image", rerun::Image({ img.height, img.width, 3 }, img.data));

	return 0;
}