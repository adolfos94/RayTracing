#ifndef BVH_H
#define BVH_H

#include "aabb.hpp"
#include "ray.hpp"
#include "hittable.hpp"
#include "hittable_list.hpp"

class bvh_node : public hittable
{
public:

  __device__ bvh_node(curandState* local_rand_state, hittable_list* list) :
    bvh_node(local_rand_state, list->objects, 0, list->num_objects) {}

  __device__ bvh_node(curandState* local_rand_state, cuda::std::array<hittable*, HITTABLES_SIZE>& objects, size_t start, size_t end)
  {
    int axis = random_int(local_rand_state, 0, 2);

    auto comparator = (axis == 0) ? box_x_compare
      : (axis == 1) ? box_y_compare
      : box_z_compare;

    size_t object_span = end - start;

    if (object_span == 1)
    {
      left = right = objects[start];
    }
    else if (object_span == 2)
    {
      left = objects[start];
      right = objects[start + 1];
    }
    else
    {
      sort(objects, start, end, comparator);

      auto mid = (start + object_span) / 2;

      left = new bvh_node(local_rand_state, objects, start, mid);
      right = new bvh_node(local_rand_state, objects, mid, end);
    }

    bbox = aabb(left->bounding_box(), right->bounding_box());
  }

  ~bvh_node()
  {
    if (left != nullptr)
      delete left;

    if (right != nullptr)
      delete right;
  }

  __device__ bool hit(const ray& r, const interval& ray_t, hit_record& rec) const override
  {
    if (!bbox.hit(r, ray_t))
      return false;

    bool hit_left = left->hit(r, ray_t, rec);
    bool hit_right = right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);

    return hit_left || hit_right;
  }

  __device__ aabb bounding_box() const override
  {
    return bbox;
  }

private:
  hittable* left = nullptr;
  hittable* right = nullptr;
  aabb bbox;

  __device__ static bool box_compare(const hittable* a, const hittable* b, int axis_index)
  {
    auto a_axis_interval = a->bounding_box().axis_interval(axis_index);
    auto b_axis_interval = b->bounding_box().axis_interval(axis_index);
    return a_axis_interval.min < b_axis_interval.min;
  }

  __device__ static bool box_x_compare(const hittable* a, const hittable* b)
  {
    return box_compare(a, b, 0);
  }

  __device__ static bool box_y_compare(const hittable* a, const hittable* b)
  {
    return box_compare(a, b, 1);
  }

  __device__ static bool box_z_compare(const hittable* a, const hittable* b)
  {
    return box_compare(a, b, 2);
  }

  __device__ void sort(cuda::std::array<hittable*, HITTABLES_SIZE>& objects, size_t begin, size_t end, bool(*compare)(const hittable* a, const hittable* b))
  {
    printf("Start: %llu\t End: %llu\n", begin, end);

    for (size_t i = begin; i < end - 1; ++i)
    {
      size_t min_idx = i;
      for (size_t j = i + 1; j < end; ++j)
      {
        if (compare(objects[j], objects[min_idx]))
          min_idx = j;
      }

      if (min_idx != i)
        cuda::std::swap(objects[i], objects[min_idx]);
    }
  }
};

#endif