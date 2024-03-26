#ifndef CAMERA_H
#define CAMERA_H

#include "utilities.hpp"

class camera
{
public:

	// Camera params
	size_t width = 1280;       		   		// Rendered image width
	size_t height = 720;               		// Rendered image height

	camera() {};

	/// <summary>
	/// Initializes viewport dimensions and camera coordinate frame.
	/// </summary>
	void initialize()
	{
	}

private:
};

#endif