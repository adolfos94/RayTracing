file(GLOB_RECURSE includes_h CONFIGURE_DEPENDS "*.h")
file(GLOB_RECURSE includes_hpp CONFIGURE_DEPENDS "*.hpp")
file(GLOB_RECURSE sources_cpp CONFIGURE_DEPENDS "*.cpp")
file(GLOB_RECURSE sources_cuda CONFIGURE_DEPENDS "*.cu")

set(SOURCES  
    ${includes_h}
    ${includes_hpp}
    ${sources_cpp}
    ${sources_cuda}
)

set(INCLUDE_DIRS 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    ${CUDA_INCLUDE_DIRS}
    ${Stb_INCLUDE_DIR}
)

set(LIBRARIES
    ${CUDA_LIBRARIES}
    cudadevrt 
    rerun_sdk
)

add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 75)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()