#include "utilities.hpp"
#include "hittables.hpp"
#include "camera.hpp"

#include "geometries/sphere.hpp"
#include "materials/lambertian.hpp"
#include "materials/metal.hpp"

int main()
{
	// ReRun visualization
	const auto rec = rerun::RecordingStream("RayTracing");
	rec.spawn().exit_on_failure();

	// **** World ****
	hittable_list world;

	auto material_ground = std::make_shared<lambertian>(color(0.8, 0.8, 0.0));
	auto material_center = std::make_shared<lambertian>(color(0.7, 0.3, 0.3));
	auto material_left = std::make_shared<metal>(color(0.8, 0.8, 0.8), 0.3);
	auto material_right = std::make_shared<metal>(color(0.8, 0.6, 0.2), 1.0);

	world.add(std::make_shared<sphere>("world/sphere_1", point3(0.0, -100.5, -1.0), 100.0, material_ground));
	world.add(std::make_shared<sphere>("world/sphere_2", point3(0.0, 0.0, -1.0), 0.5, material_center));
	world.add(std::make_shared<sphere>("world/sphere_3", point3(-1.0, 0.0, -1.0), 0.5, material_left));
	world.add(std::make_shared<sphere>("world/sphere_4", point3(1.0, 0.0, -1.0), 0.5, material_right));

	// **** Camera ****
	camera cam = camera(1280, 720);
	cam.render(world, rec);

	return 0;
}