#ifndef INTERVAL_H
#define INTERVAL_H

#include "utilities.hpp"

class interval
{
public:
	double min, max;

	// Default interval is empty
	__device__ interval() : min(+infinity), max(-infinity) {}

	__device__ interval(double _min, double _max) : min(_min), max(_max) {}

	// Create the interval tightly enclosing the two input intervals.
	__device__ interval(const interval& a, const interval& b)
	{
		min = a.min <= b.min ? a.min : b.min;
		max = a.max >= b.max ? a.max : b.max;
	}

	__device__ bool contains(double x) const
	{
		return min <= x && x <= max;
	}

	__device__ bool surrounds(double x) const
	{
		return min < x && x < max;
	}

	__device__ double clamp(double x) const
	{
		if (x < min) return min;
		if (x > max) return max;
		return x;
	}

	__device__ interval expand(double delta) const
	{
		auto padding = delta / 2.0;
		return interval(min - padding, max + padding);
	}
};

#endif