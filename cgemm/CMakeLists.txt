cmake_minimum_required(VERSION 3.28)
project(cgemm LANGUAGES C CUDA)

# set(CMAKE_BUILD_TYPE Debug)

find_package(Python REQUIRED COMPONENTS Development)

add_subdirectory(src)

target_link_libraries(cgemm PRIVATE Python::Python)

set_target_properties(cgemm PROPERTIES 
    PREFIX ""
    CUDA_SEPARABLE_COMPILATION ON
)

