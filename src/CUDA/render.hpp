#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "color.hpp"
#include "camera.hpp"


namespace render
{
	namespace cuda
	{
		constexpr size_t BLOCK_SIZE = 32; // Thread block size

		void get_device_params();

		void render(camera& cam, image& img);
	} // namespace cuda
} // namespace render