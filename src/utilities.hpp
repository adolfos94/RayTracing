#pragma once

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <rerun.hpp>
#include <spdlog/spdlog.h>

// Structs
struct image
{
	size_t width = 0;
	size_t height = 0;
	size_t size = 0;
	uint8_t* data = nullptr;

	image() {};

	image(const size_t w, const size_t h, uint8_t val = 0) : width(w), height(h)
	{
		size = width * height * 3;
		data = new uint8_t[size]();
	}
};