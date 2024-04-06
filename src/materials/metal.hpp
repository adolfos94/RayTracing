#ifndef METAL_H
#define METAL_H

#include "material.hpp"

class metal : public material
{
public:
	__device__  metal(const color& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

	__device__ bool scatter(
		const ray& r_in,
		const hit_record& rec,
		vec3& attenuation,
		ray& scattered,
		curandState* local_rand_state) const override
	{
		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);

		scattered = ray(rec.p, reflected + fuzz * random_unit_vector(local_rand_state));
		attenuation = albedo;

		return (dot(scattered.direction(), rec.normal) > 0);
	}

private:
	vec3 albedo;
	double fuzz;
};

#endif