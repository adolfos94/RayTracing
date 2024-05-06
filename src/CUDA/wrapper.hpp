#pragma once

#include "color.hpp"
#include "camera.hpp"
#include "hittable_list.hpp"

#include "geometries/bvh.hpp"
#include "geometries/sphere.hpp"
#include "materials/lambertian.hpp"
#include "materials/metal.hpp"
#include "materials/dielectric.hpp"

#define checkCudaErrors(val) cuda::check_cuda( (val), #val, __FILE__, __LINE__ )

namespace cuda
{
  constexpr size_t BLOCK_SIZE = 16; // Thread block size

  inline void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
  {
    if (result != cudaSuccess)
    {
      spdlog::error("[CUDA] Error: {} at {}:{} '{}'",
        cudaGetErrorString(result), file, line, func);

      cudaDeviceReset();
      exit(-1);
    }
  }
}