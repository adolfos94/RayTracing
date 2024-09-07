#ifndef IMAGE_H
#define IMAGE_H

// Disable strict warnings for this header from the Microsoft Visual C++ compiler.
#ifdef _MSC_VER
#pragma warning (push, 0)
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG

#include "external/stb_image.h"
#include "material.hpp"

class image
{
public:
  image() {}

  // Loads image data from the specified file.
  image(const char* image_filename)
  {
    auto filename = std::string(image_filename);
    auto imagedir = getenv("IMAGES");

    // Hunt for the image file in some likely locations.
    if (imagedir && load(std::string(imagedir) + "/" + image_filename)) return;
    if (load(filename)) return;
    if (load("images/" + filename)) return;
    if (load("../images/" + filename)) return;
    if (load("../../images/" + filename)) return;
    if (load("../../../images/" + filename)) return;
    if (load("../../../../images/" + filename)) return;
    if (load("../../../../../images/" + filename)) return;
    if (load("../../../../../../images/" + filename)) return;
  }

  image(const int width, const int height) : image_width(width), image_height(height)
  {
    size_t total_bytes = image_width * image_height * bytes_per_pixel;
    bytes_per_scanline = image_width * bytes_per_pixel;
    bdata = new uint8_t[total_bytes]();
  }

  //~image()
  //{
  //  if (bdata == nullptr)
  //    return;

  //  delete[] bdata;
  //  STBI_FREE(fdata);
  //}

  // Loads the linear (gamma=1) image data from the given file name. Returns true if the
  // load succeeded. The resulting data buffer contains the three [0.0, 1.0]
  // floating-point values for the first pixel (red, then green, then blue). Pixels are
  // contiguous, going left to right for the width of the image, followed by the next row
  // below, for the full height of the image
  bool load(const std::string& filename)
  {
    auto n = bytes_per_pixel; // Dummy out parameter: original components per pixel
    fdata = stbi_loadf(filename.c_str(), &image_width, &image_height, &n, bytes_per_pixel);

    if (fdata == nullptr)
      return false;

    bytes_per_scanline = image_width * bytes_per_pixel;
    convert_to_bytes();
    return true;
  }

  void save(const std::string& filename)
  {
    std::ofstream output_file(filename);

    if (!output_file.is_open())
      return;

    output_file << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++)
    {
      for (int i = 0; i < image_width; i++)
      {
        auto idx = j * image_width + i;

        int ir = bdata[idx * bytes_per_pixel + 0];
        int ig = bdata[idx * bytes_per_pixel + 1];
        int ib = bdata[idx * bytes_per_pixel + 2];

        output_file << ir << ' ' << ig << ' ' << ib << '\n';
      }
    }
    output_file.close();
  }

  __device__ __host__ int width() const { return (bdata == nullptr) ? 0 : image_width; } // height
  __device__ __host__ int height() const { return (bdata == nullptr) ? 0 : image_height; } // width
  __device__ __host__ int size() const { return (bdata == nullptr) ? 0 : image_width * image_height * bytes_per_pixel; } // size in bytes

  void set_width(int width) { image_width = (bdata == nullptr) ? 0 : width; }
  void set_height(int height) { image_height = (bdata == nullptr) ? 0 : height; }

  // Return the address of the three RGB bytes of the pixel at x,y. If there is no image
  // data, returns magenta.
  __device__ const uint8_t* pixel_data(int x, int y) const
  {
    static uint8_t magenta[] = { 255, 0, 255 };

    if (bdata == nullptr)
      return magenta;

    x = clamp(x, 0, image_width);
    y = clamp(y, 0, image_height);

    return bdata + y * bytes_per_scanline + x * bytes_per_pixel;
  }

  __device__ void set_pixel_data(int x, int y, uint8_t r, uint8_t g, uint8_t b)
  {
    if (bdata == nullptr)
      return;

    x = clamp(x, 0, image_width);
    y = clamp(y, 0, image_height);

    auto idx = y * image_width + x;

    bdata[idx * bytes_per_pixel + 0] = r;
    bdata[idx * bytes_per_pixel + 1] = g;
    bdata[idx * bytes_per_pixel + 2] = b;
  }

  const uint8_t* data() const
  {
    return bdata;
  }

  uint8_t*& data()
  {
    return bdata;
  }

private:
  const int bytes_per_pixel = 3;
  float* fdata = nullptr; // Linear floating point pixel data
  uint8_t* bdata = nullptr; // Linear 8-bit pixel data
  int image_width = 0; // Loaded image width
  int image_height = 0; // Loaded image height
  int bytes_per_scanline = 0;

  // Return the value clamped to the range [low, high).
  __device__ static int clamp(int x, int low, int high)
  {
    if (x < low) return low;
    if (x < high) return x;
    return high - 1;
  }

  static uint8_t float_to_byte(float value)
  {
    if (value <= 0.0f)
      return 0;
    if (1.0f <= value)
      return 255;
    return static_cast<uint8_t>(256.0f * value);
  }

  // Convert the linear floating point pixel data to bytes, storing the resulting byte
  // data in the `bdata` member.
  void convert_to_bytes()
  {
    size_t total_bytes = image_width * image_height * bytes_per_pixel;
    bdata = new uint8_t[total_bytes];

    // Iterate through all pixel components, converting from [0.0, 1.0] float values to
    // unsigned [0, 255] byte values.
    auto* bptr = bdata;
    auto* fptr = fdata;
    for (auto i = 0; i < total_bytes; i++, fptr++, bptr++)
      *bptr = float_to_byte(*fptr);
  }
};

// Restore MSVC compiler warnings
#ifdef _MSC_VER
#pragma warning (pop)
#endif

#endif