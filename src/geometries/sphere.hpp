#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.hpp"

class sphere : public hittable
{
public:
  // Stationary sphere
  __device__ sphere(const point3& _center, double _radius, material* _mat) :
    center1(_center), radius(fmax(0.0, _radius)), mat(_mat), is_moving(false)
  {
    auto rvec = vec3(radius, radius, radius);
    bbox = aabb(_center - rvec, _center + rvec);
  }

  // Moving sphere
  __device__ sphere(const point3& _center1, const point3& _center2, double _radius, material* _mat) :
    center1(_center1), radius(fmax(0.0, _radius)), mat(_mat), is_moving(true)
  {
    auto rvec = vec3(radius, radius, radius);
    aabb box1(_center1 - rvec, _center1 + rvec);
    aabb box2(_center2 - rvec, _center2 + rvec);
    bbox = aabb(box1, box2);

    center_vec = _center2 - _center1;
  }

  __device__ bool hit(const ray& r, const interval& ray_t, hit_record& rec) const override
  {
    point3 center = is_moving ? sphere_center(r.time()) : center1;
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
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat = mat;

    return true;
  }

  __device__ aabb bounding_box() const override
  {
    return bbox;
  }

private:
  point3 center1;
  double radius;
  material* mat;
  bool is_moving;
  vec3 center_vec;
  aabb bbox;

  // Linearly interpolate from center1 to center2 according to time, where t=0 yields
  // center1, and t=1 yields center2.
  __device__ point3 sphere_center(double time) const
  {
    return center1 + time * center_vec;
  }
};

#endif