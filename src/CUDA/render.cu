#include "render.hpp"

__global__ void render_kernel(camera cam, image img)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i >= img.width || j >= img.height)
		return;

	ray ray = cam.get_ray(i, j);
	color pixel_color = ray_color(ray);

	sample_color(pixel_color);

	img.data[(j * img.width + i) * 3 + 0] = static_cast<uint8_t>(pixel_color.x());
	img.data[(j * img.width + i) * 3 + 1] = static_cast<uint8_t>(pixel_color.y());
	img.data[(j * img.width + i) * 3 + 2] = static_cast<uint8_t>(pixel_color.z());
}

void render::cuda::render(camera& cam, image& img)
{
	dim3 blocks(img.width / BLOCK_SIZE + 1, img.height / BLOCK_SIZE + 1);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	spdlog::info("[CUDA::Render] Exec kernel <<< {}, {} | {} >>>", blocks.x, blocks.y, threads.x * threads.y);

	render_kernel << <blocks, threads >> > (cam, img);
}