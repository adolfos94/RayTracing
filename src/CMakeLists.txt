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
)

set(LIBRARIES
    ${CUDA_LIBRARIES}
    spdlog::spdlog
    rerun_sdk
)

add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})