#include "color.hpp"
#include "camera.hpp"
#include "hittable_list.hpp"
#include "geometries/sphere.hpp"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

constexpr size_t BLOCK_SIZE = 32; // Thread block size

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result != cudaSuccess)
	{
		spdlog::error("[CUDA] Error: {} at {}:{} '{}'",
			cudaGetErrorString(result), file, line, func);

		cudaDeviceReset();
		exit(-1);
	}
}

__global__ void world_kernel(hittable_list** d_world, size_t num_objects)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		*d_world = new hittable_list();
		(*d_world)->num_objects = num_objects;
		(*d_world)->objects = new hittable * [num_objects]();
		(*d_world)->objects[0] = new sphere(vec3(0, 0, -1), 0.5);
		(*d_world)->objects[1] = new sphere(vec3(0, -100.5, -1), 100);
	}
}

__global__ void camera_kernel(camera** d_camera, size_t width, size_t height)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		*d_camera = new camera(width, height);
	}
}

__global__ void render_kernel(hittable_list** d_world, camera** d_camera, image d_image)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i >= d_image.width || j >= d_image.height)
		return;

	ray ray = (*d_camera)->get_ray(i, j);
	color pixel_color = ray_color(ray, d_world);

	sample_color(pixel_color);

	d_image.data[(j * d_image.width + i) * 3 + 0] = static_cast<uint8_t>(pixel_color.x());
	d_image.data[(j * d_image.width + i) * 3 + 1] = static_cast<uint8_t>(pixel_color.y());
	d_image.data[(j * d_image.width + i) * 3 + 2] = static_cast<uint8_t>(pixel_color.z());
}

int main()
{
	// ReRun visualization
	const auto rec = rerun::RecordingStream("RayTracing");
	rec.spawn().exit_on_failure();

	// Create world with CUDA
	hittable_list** d_world;
	checkCudaErrors(cudaMalloc(&d_world, sizeof(hittable_list**)));
	world_kernel << <1, 1 >> > (d_world, 2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Create camera
	size_t width = 1280;
	size_t height = 720;
	camera** d_camera;
	checkCudaErrors(cudaMalloc(&d_camera, sizeof(camera**)));
	camera_kernel << <1, 1 >> > (d_camera, width, height);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Create image
	image h_image = image(width, height);

	image d_image;
	d_image.width = h_image.width;
	d_image.height = h_image.height;
	checkCudaErrors(cudaMalloc(&d_image.data, h_image.size));

	// Render scene with CUDA
	dim3 blocks(width / BLOCK_SIZE + 1, height / BLOCK_SIZE + 1);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	render_kernel << <blocks, threads >> > (d_world, d_camera, d_image);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(h_image.data, d_image.data, h_image.size, cudaMemcpyDeviceToHost));

	rec.log_timeless("RayTracing", rerun::Image({ h_image.height, h_image.width, 3 }, h_image.data));

	//cudaFree(d_img.data);

	return 0;
}