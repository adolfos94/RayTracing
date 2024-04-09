#ifndef RAY_H
#define RAY_H

#include "vec3.hpp"

class ray
{
public:
	__device__ ray() {}

	__device__ ray(const point3& origin, const vec3& direction) :
		orig(origin), dir(direction), tm(0.0) {}

	__device__ ray(const point3& origin, const vec3& direction, double time) :
		orig(origin), dir(direction), tm(time) {}

	__device__ point3 origin() const { return orig; }
	__device__ vec3 direction() const { return dir; }

	__device__ point3 at(double t) const
	{
		return orig + t * dir;
	}
	__device__ double time() const
	{
		return tm;
	}

private:
	point3 orig;
	vec3 dir;
	double tm;
};

#endif