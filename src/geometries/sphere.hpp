#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.hpp"
#include "vec3.hpp"

class sphere : public hittable
{
public:
	sphere(const std::string& _name, const point3 _center, const double _radius) :
		name(_name), center(_center), radius(_radius) {}

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

		return true;
	}

	void render(const rerun::RecordingStream& rec) const override
	{
		rec.log_timeless(
			name,
			rerun::Asset3D::from_file("./assets/sphere.obj").
			value_or_throw());

		const auto trs =
			rerun::TranslationRotationScale3D({
			(float)center.x(),
			(float)center.y(),
			(float)center.z() },
			(float)radius);

		rec.log_timeless(
			name,
			rerun::Collection<rerun::OutOfTreeTransform3D>(rerun::OutOfTreeTransform3D(trs)));
	}

private:

	std::string name;
	point3 center;
	double radius;
};

#endif