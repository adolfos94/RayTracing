#include "render.hpp"

void GetDeviceInfo()
{
	int driverVersion;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cudaDriverGetVersion(&driverVersion);

	spdlog::info("[CUDA] Driver version: {}", driverVersion);
	spdlog::info("[CUDA] Device name: {}", prop.name);
	spdlog::info("[{}] Memory Clock Rate (KHz): {}", prop.name, prop.memoryClockRate);
	spdlog::info("[{}] Memory Bus Width (bits): {}", prop.name, prop.memoryBusWidth);
	spdlog::info("[{}] Peak Memory Bandwidth (GB/s): {}", prop.name, 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
	spdlog::info("[{}] Max Threads per Block: {}", prop.name, prop.maxThreadsPerBlock);
}

void render::cuda::get_device_params()
{
	GetDeviceInfo();
}