#include "utilities.hpp"
#include "hittables.hpp"
#include "camera.hpp"

#include "geometries/sphere.hpp"

int main()
{
	// ReRun visualization
	const auto rec = rerun::RecordingStream("RayTracing");
	rec.spawn().exit_on_failure();

	// **** World ****
	hittable_list world;
	world.add(std::make_shared<sphere>("world/sphere_1", point3(0, 0, -1), 0.5));
	world.add(std::make_shared<sphere>("world/sphere_2", point3(0, -100.5, -1), 100));

	// **** Camera ****
	camera cam = camera(1280, 720);
	cam.render(world, rec);

	return 0;
}