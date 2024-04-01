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

	__device__ bool hit(const ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const
	{
		hit_record temp_rec;
		bool hit_anything = false;
		auto closest_so_far = ray_tmax;

		for (size_t i = 0; i < num_objects; ++i)
		{
			if (objects[i]->hit(r, ray_tmin, closest_so_far, temp_rec))
			{
				hit_anything = true;
				closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}

		return hit_anything;
	}
};

#endif