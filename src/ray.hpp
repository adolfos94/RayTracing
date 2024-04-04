#ifndef RAY_H
#define RAY_H

#include "vec3.hpp"

constexpr size_t MAX_DEPTH = 50;   // Maximum number of ray bounces into scene

class ray
{
public:
	__device__ ray() {}

	__device__ ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {}

	__device__ point3 origin() const { return orig; }
	__device__ vec3 direction() const { return dir; }

	__device__ point3 at(double t) const
	{
		return orig + t * dir;
	}

private:
	point3 orig;
	vec3 dir;
};

#endif