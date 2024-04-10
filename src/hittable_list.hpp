#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.hpp"

class hittable_list : public hittable
{
public:

	hittable** objects = nullptr;
	size_t num_objects = 0;

	__device__ hittable_list(size_t N) : num_objects(N)
	{
		objects = new hittable * [num_objects]();
	}

	__device__ void add(hittable* object)
	{
		static size_t idx = 0;

		if (idx >= num_objects)
			return;

		objects[idx++] = object;

		bbox = aabb(bbox, object->bounding_box());
	}

	__device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const
	{
		hit_record temp_rec;
		bool hit_anything = false;
		auto closest_so_far = ray_t.max;

		for (size_t i = 0; i < num_objects; ++i)
		{
			if (!objects[i])
				continue;

			if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec))
			{
				hit_anything = true;
				closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}

		return hit_anything;
	}

	__device__ aabb bounding_box() const override
	{
		return bbox;
	}
private:
	aabb bbox;
};

#endif