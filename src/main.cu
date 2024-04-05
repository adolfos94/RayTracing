#include "cuda/wrapper.hpp"

__global__ void world_kernel(hittable_list** d_world, size_t num_objects)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		*d_world = new hittable_list(num_objects);

		auto material_ground = new lambertian(color(0.8, 0.8, 0.0));
		auto material_center = new lambertian(color(0.1, 0.2, 0.5));
		auto material_left = new dielectric(1.5);
		auto material_right = new metal(color(0.8, 0.6, 0.2), 0.0);

		(*d_world)->objects[0] = new sphere(point3(0.0, -100.5, -1.0), 100.0, material_ground);
		(*d_world)->objects[1] = new sphere(point3(0.0, 0.0, -1.0), 0.5, material_center);
		(*d_world)->objects[2] = new sphere(point3(-1.0, 0.0, -1.0), 0.5, material_left);
		(*d_world)->objects[3] = new sphere(point3(-1.0, 0.0, -1.0), -0.4, material_left);
		(*d_world)->objects[4] = new sphere(point3(1.0, 0.0, -1.0), 0.5, material_right);
	}
}

__global__ void camera_kernel(camera** d_camera, size_t width, size_t height)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		*d_camera = new camera(width, height);

		(*d_camera)->vfov = 20;
		(*d_camera)->lookfrom = point3(13, 2, 3);
		(*d_camera)->lookat = point3(0, 0, 0);

		(*d_camera)->defocus_angle = 0.6;
		(*d_camera)->focus_dist = 10.0;

		(*d_camera)->initialize();
	}
}

__global__ void random_kernel(curandState* state, size_t width, size_t height)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i >= width || j >= height)
		return;

	int idx = j * width + i;

	curand_init(1234, idx, 0, &state[idx]);
}

__global__ void render_kernel(hittable_list** d_world, camera** d_camera, curandState* rand_state, image d_image)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i >= d_image.width || j >= d_image.height)
		return;

	int idx = j * d_image.width + i;
	curandState local_rand_state = rand_state[idx];

	color pixel_color(0, 0, 0);
	for (size_t s = 0; s < SAMPLES_PER_PIXEL; ++s)
	{
		ray ray = (*d_camera)->get_ray(&local_rand_state, i, j);
		pixel_color += ray_color(ray, d_world, &local_rand_state);
	}
	sample_color(SAMPLES_PER_PIXEL, pixel_color);

	rand_state[idx] = local_rand_state;

	d_image.data[idx * 3 + 0] = static_cast<uint8_t>(pixel_color.x());
	d_image.data[idx * 3 + 1] = static_cast<uint8_t>(pixel_color.y());
	d_image.data[idx * 3 + 2] = static_cast<uint8_t>(pixel_color.z());
}

int main()
{
	// ReRun visualization
	const auto rec = rerun::RecordingStream("RayTracing");
	rec.spawn().exit_on_failure();

	// Create world with CUDA
	hittable_list** d_world;
	checkCudaErrors(cudaMalloc(&d_world, sizeof(hittable_list**)));
	world_kernel << <1, 1 >> > (d_world, 5);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Create camera
	size_t width = 1280 * 1;
	size_t height = 720 * 1;
	camera** d_camera;
	checkCudaErrors(cudaMalloc(&d_camera, sizeof(camera**)));
	camera_kernel << <1, 1 >> > (d_camera, width, height);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Render scene with CUDA
	dim3 blocks(width / cuda::BLOCK_SIZE + 1, height / cuda::BLOCK_SIZE + 1);
	dim3 threads(cuda::BLOCK_SIZE, cuda::BLOCK_SIZE);

	// Create random state
	curandState* d_rand_state;
	checkCudaErrors(cudaMalloc(&d_rand_state, width * height * sizeof(curandState)));
	random_kernel << <blocks, threads >> > (d_rand_state, width, height);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Create image
	image h_image = image(width, height);

	image d_image;
	d_image.width = h_image.width;
	d_image.height = h_image.height;
	checkCudaErrors(cudaMalloc(&d_image.data, h_image.size));

	render_kernel << <blocks, threads >> > (d_world, d_camera, d_rand_state, d_image);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(h_image.data, d_image.data, h_image.size, cudaMemcpyDeviceToHost));

	rec.log_timeless("RayTracing", rerun::Image({ h_image.height, h_image.width, 3 }, h_image.data));

	//cudaFree(d_img.data);

	return 0;
}