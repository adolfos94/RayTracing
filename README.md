<center>

# Ray Tracing on CUDA

![CMake](https://img.shields.io/badge/CMake-%23064F8C?style=for-the-badge&logo=cmake)
![C++](https://img.shields.io/badge/C%2B%2B-%2300599C?style=for-the-badge&logo=cplusplus)
![CUDA](https://img.shields.io/badge/CUDA-%2376B900?style=for-the-badge&logo=nvidia&logoColor=white)

This is my RayTracing framework based on **Ray Tracing in One Weekend** series of books on CUDA and modern C++.

<img width=720 src="doc/result.png"/>

</center>

## Building and Running

Copies of the source are provided for you to check your work and compare against. If you wish to build
the provided source, this project uses CMake. To build, go to the root of the project directory and
run the following commands to create the debug version of every executable:

    $ cmake -B build
    $ cmake --build build

## Running The Programs

On Linux, from the terminal, run like this:

    $ build/RayTracing

On Windows, run like this:

    build\debug\RayTracing

## Requirements
### CUDA
- **Install**: [CUDA](https://developer.nvidia.com/cuda-toolkit)

### ReRun

- **C++**: [Guide](https://www.rerun.io/docs/getting-started/cpp)
- **Python**: `pip install rerun-sdk` or on [`conda`](https://github.com/conda-forge/rerun-sdk-feedstock)
- **Rust**: `cargo add rerun`

## Future Development

- Complete book series.
  - [_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
  - [_Ray Tracing: The Next Week_](https://raytracing.github.io/books/RayTracingTheNextWeek.html)
