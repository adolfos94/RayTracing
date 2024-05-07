#include "cuda/wrapper.hpp"

__global__ void world_kernel(hittable_list** d_world)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    *d_world = new hittable_list();

    auto ground_material = new lambertian(color(0.5, 0.5, 0.5));
    (*d_world)->add(new sphere(point3(0, -1000, 0), 1000, ground_material));

    curandState local_rand_state;
    curand_init(1234, 0, 0, &local_rand_state);

    for (int a = -11; a < 11; a++)
    {
      for (int b = -11; b < 11; b++)
      {
        auto choose_mat = random_double(&local_rand_state);
        point3 center(a + 0.9 * random_double(&local_rand_state), 0.2, b + 0.9 * random_double(&local_rand_state));

        if ((center - point3(4, 0.2, 0)).length() > 0.9)
        {
          if (choose_mat < 0.8)
          {
            // diffuse
            auto albedo = color::random(&local_rand_state) * color::random(&local_rand_state);
            auto sphere_material = new lambertian(albedo);

            // moving spheres
            auto center2 = center + vec3(0, random_double(&local_rand_state, 0.0, 0.5), 0);
            (*d_world)->add(new sphere(center, center2, 0.2, sphere_material));
          }
          else if (choose_mat < 0.95)
          {
            // metal
            auto albedo = color::random(&local_rand_state, 0.5, 1);
            auto fuzz = random_double(&local_rand_state, 0, 0.5);
            auto sphere_material = new metal(albedo, fuzz);
            (*d_world)->add(new sphere(center, 0.2, sphere_material));
          }
          else
          {
            // glass
            auto sphere_material = new dielectric(1.5);
            (*d_world)->add(new sphere(center, 0.2, sphere_material));
          }
        }
      }
    }

    auto material1 = new dielectric(1.5);
    (*d_world)->add(new sphere(point3(0, 1, 0), 1.0, material1));

    auto material2 = new lambertian(color(0.4, 0.2, 0.1));
    (*d_world)->add(new sphere(point3(-4, 1, 0), 1.0, material2));

    auto material3 = new metal(color(0.7, 0.6, 0.5), 0.0);
    (*d_world)->add(new sphere(point3(4, 1, 0), 1.0, material3));

    //*d_world = new hittable_list(new bvh_node(&local_rand_state, *d_world));
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
    ray ray = (*d_camera)->get_ray(i, j, &local_rand_state);
    pixel_color += (*d_camera)->ray_color(ray, d_world, &local_rand_state);
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

  // Get the current stack size.
  size_t stackSize;
  cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
  cudaDeviceSetLimit(cudaLimitStackSize, stackSize << 1);

  // Create world with CUDA
  hittable_list** d_world;
  checkCudaErrors(cudaMalloc(&d_world, sizeof(hittable_list**)));
  world_kernel << <1, 1 >> > (d_world);
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