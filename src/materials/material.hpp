#ifndef MATERIAL_H
#define MATERIAL_H

#include "ray.hpp"
#include "color.hpp"

class hit_record;

class material
{
public:
	__device__ virtual ~material() = default;

	__device__ virtual bool scatter(
		const ray& r_in,
		const hit_record& rec,
		vec3& attenuation,
		ray& scattered,
		curandState* local_rand_state) const = 0;
};

#endif