#ifndef COLOR_H
#define COLOR_H

#include "vec3.hpp"

using color = vec3;

__device__ inline void sample_color(color& pixel_color)
{
	// Write the translated [0,255] value of each color component.
	pixel_color[0] = 255.999 * pixel_color.x();
	pixel_color[1] = 255.999 * pixel_color.y();
	pixel_color[2] = 255.999 * pixel_color.z();
}

#endif