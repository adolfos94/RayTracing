#include "render.hpp"

__global__ void render_kernel(image img)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	img.data[(j * img.width + i) * 3 + 0] = static_cast<uint8_t>(255);
	img.data[(j * img.width + i) * 3 + 1] = static_cast<uint8_t>(0);
	img.data[(j * img.width + i) * 3 + 2] = static_cast<uint8_t>(0);
}

void render::cuda::render(image& img)
{
	dim3 blocks(img.width / BLOCK_SIZE + 1, img.height / BLOCK_SIZE + 1);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	render_kernel << <blocks, threads >> > (img);
}