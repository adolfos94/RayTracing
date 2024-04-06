#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "material.hpp"

class lambertian : public material
{
public:
	__device__ lambertian(const vec3& a) : albedo(a) {}

	__device__ bool scatter(
		const ray& r_in,
		const hit_record& rec,
		vec3& attenuation,
		ray& scattered,
		curandState* local_rand_state) const override
	{
		auto scatter_direction = rec.normal + random_unit_vector(local_rand_state);

		// Catch degenerate scatter direction
		if (scatter_direction.near_zero())
			scatter_direction = rec.normal;

		scattered = ray(rec.p, scatter_direction);
		attenuation = albedo;

		return true;
	}

private:
	vec3 albedo;
};

#endif