#ifndef DIELECTRIC_H
#define DIELECTRIC_H

#include "material.hpp"

class dielectric : public material
{
public:
	__device__ dielectric(double index_of_refraction) : ir(index_of_refraction) {}

	__device__ bool scatter(
		const ray& r_in,
		const hit_record& rec,
		vec3& attenuation,
		ray& scattered,
		curandState* local_rand_state) const override
	{
		attenuation = color(1.0, 1.0, 1.0);
		double refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

		vec3 unit_direction = unit_vector(r_in.direction());
		double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
		double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

		bool cannot_refract = refraction_ratio * sin_theta > 1.0;
		vec3 direction;
		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double(local_rand_state))
			direction = reflect(unit_direction, rec.normal);
		else
			direction = refract(unit_direction, rec.normal, refraction_ratio);

		scattered = ray(rec.p, direction);

		return true;
	}

private:
	double ir; // Index of Refraction

	// Use Schlick's approximation for reflectance.
	__device__ static double reflectance(double cosine, double ref_idx)
	{
		auto r0 = (1 - ref_idx) / (1 + ref_idx);
		r0 = r0 * r0;
		return r0 + (1 - r0) * pow((1 - cosine), 5);
	}
};

#endif