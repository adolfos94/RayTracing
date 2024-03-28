#include "cuda/render.hpp"

int main()
{
	// ReRun visualization
	const auto rec = rerun::RecordingStream("RayTracing");
	rec.spawn().exit_on_failure();

	camera cam = camera(1280, 720);

	image img = image(cam.image_width, cam.image_height);

	// Render scene with CUDA
	image d_img;
	d_img.width = img.width;
	d_img.height = img.height;
	cudaMalloc(&d_img.data, img.size);

	render::cuda::get_device_params();
	render::cuda::render(cam, d_img);
	cudaMemcpy(img.data, d_img.data, img.size, cudaMemcpyDeviceToHost);

	rec.log_timeless("RayTracing", rerun::Image({ img.height, img.width, 3 }, img.data));

	cudaFree(d_img.data);

	return 0;
}