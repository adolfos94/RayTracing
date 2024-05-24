#ifndef TEXTURE_H
#define TEXTURE_H

#include "material.hpp"

class texture 
{
public:
  __device__ virtual ~texture() = default;

  __device__ virtual color value(double u, double v, const point3& p) const = 0;
};

class solid_color : public texture 
{
public:
  __device__ solid_color(const color& albedo) : albedo(albedo) {}

  __device__ solid_color(double red, double green, double blue) : solid_color(color(red, green, blue)) {}

  __device__ color value(double u, double v, const point3& p) const override
  {
    return albedo;
  }

private:
  color albedo;
};

class checker_texture : public texture 
{
public:
  __device__ checker_texture(double scale, texture* even, texture* odd)
    : inv_scale(1.0 / scale), even(even), odd(odd) {}

  __device__ checker_texture(double scale, const color& c1, const color& c2)
    : inv_scale(1.0 / scale), even(new solid_color(c1)), odd(new solid_color(c2)) {}

  __device__ color value(double u, double v, const point3& p) const override
  {
    auto xInteger = int(std::floor(inv_scale * p.x()));
    auto yInteger = int(std::floor(inv_scale * p.y()));
    auto zInteger = int(std::floor(inv_scale * p.z()));

    bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;

    return isEven ? even->value(u, v, p) : odd->value(u, v, p);
  }

private:
  double inv_scale;
  texture* even = nullptr;
  texture* odd = nullptr;
};

#endif