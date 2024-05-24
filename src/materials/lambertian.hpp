#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "material.hpp"
#include "texture.hpp"

class lambertian : public material
{
public:
	__device__ lambertian(const color& albedo) : tex(new solid_color(albedo)) {}
	__device__ lambertian(texture* tex) : tex(tex) {}

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

		scattered = ray(rec.p, scatter_direction, r_in.time());
		attenuation = tex->value(rec.u, rec.v, rec.p);

		return true;
	}

private:
	texture* tex;
};

#endif