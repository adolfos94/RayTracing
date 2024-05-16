#ifndef BVH_H
#define BVH_H

#include "aabb.hpp"
#include "ray.hpp"
#include "hittable.hpp"
#include "hittable_list.hpp"

class bvh_node : public hittable
{
public:

  __device__ bvh_node() {}

  __device__ bvh_node(hittable_list* list)
  {
    build_bst(list);
  }

  __device__ void build_bst(hittable_list* list)
  {
    auto bvh_nodes = cuda::Stack<bvh_node*>();
    bvh_nodes.push(this);

    auto indexes = cuda::Stack<cuda::std::tuple<size_t, size_t>>();
    indexes.push({ 0, list->num_objects });

    // 3D Objects list
    auto& objects = list->objects;

    while (!bvh_nodes.empty())
    {
      auto node = bvh_nodes.top();
      bvh_nodes.pop();

      auto [start, end] = indexes.top();
      indexes.pop();

      // Build the bounding box of the span of source objects
      for (size_t object_index = start; object_index < end; object_index++)
      {
        node->bbox = aabb(node->bbox, objects[object_index]->bounding_box());
      }

      auto axis = node->bbox.longest_axis();

      auto comparator = (axis == 0) ? box_x_compare
        : (axis == 1) ? box_y_compare
        : box_z_compare;

      size_t object_span = end - start;

      if (object_span == 1)
      {
        node->left = node->right = objects[start];
      }
      else if (object_span == 2)
      {
        node->left = objects[start];
        node->right = objects[start + 1];
      }
      else
      {
        sort(objects, start, end, comparator);

        auto mid = start + object_span / 2;

        node->left = new bvh_node();
        bvh_nodes.push((bvh_node*)node->left);
        indexes.push({ start, mid });

        node->right = new bvh_node();
        bvh_nodes.push((bvh_node*)node->right);
        indexes.push({ mid, end });
      }
    }
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