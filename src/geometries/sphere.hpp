#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.hpp"
#include "vec3.hpp"

class sphere : public hittable
{
public:
	sphere(
		const point3 _center,
		const double _radius,
		std::shared_ptr<material> _material,
		std::optional<std::string> _name = std::nullopt) :
		center(_center), radius(_radius), mat(_material)
	{
		if (_name.has_value())
			name = _name.value();
	}

	bool hit(const ray& r, const interval& ray_t, hit_record& rec) const override
	{
		vec3 oc = r.origin() - center;
		auto a = r.direction().length_squared();
		auto half_b = dot(oc, r.direction());
		auto c = oc.length_squared() - radius * radius;

		auto discriminant = half_b * half_b - a * c;
		if (discriminant < 0)
			return false;

		auto sqrtd = sqrt(discriminant);

		// Find the nearest root that lies in the acceptable range.
		auto root = (-half_b - sqrtd) / a;
		if (!ray_t.surrounds(root))
		{
			root = (-half_b + sqrtd) / a;
			if (!ray_t.surrounds(root))
				return false;
		}

		rec.t = root;
		rec.p = r.at(rec.t);
		rec.set_face_normal(r, (rec.p - center) / radius);
		rec.mat = mat;

		return true;
	}

private:

	point3 center;
	double radius;
	std::shared_ptr<material> mat;
	std::string name;
};

#endif
