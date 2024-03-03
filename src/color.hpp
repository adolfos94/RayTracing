#ifndef COLOR_H
#define COLOR_H

#include "vec3.hpp"

using color = vec3;

inline uint8_t r(color pixel_color)
{
	return static_cast<uint8_t>(255.999 * pixel_color.x());
}

inline uint8_t g(color pixel_color)
{
	return static_cast<uint8_t>(255.999 * pixel_color.y());
}

inline uint8_t b(color pixel_color)
{
	return static_cast<uint8_t>(255.999 * pixel_color.z());
}

#endif