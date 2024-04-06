#ifndef INTERVAL_H
#define INTERVAL_H

#include "utilities.hpp"

class interval
{
public:
	double min, max;

	__device__ interval() : min(+infinity), max(-infinity) {} // Default interval is empty

	__device__ interval(double _min, double _max) : min(_min), max(_max) {}

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
};

#endif