cmake_minimum_required(VERSION 3.28)

set(CMAKE_CUDA_ARCHITECTURES "native")
project(cgemm LANGUAGES C CUDA)

# set(CMAKE_BUILD_TYPE Debug)
find_package(Python REQUIRED COMPONENTS Development)

add_subdirectory(src)

target_compile_options(cgemm PRIVATE
    -mavx2
    -mfma
    -march=native
    -O3
    -g
    -pg
)
target_link_options(cgemm PRIVATE
    -pg
)

target_link_libraries(cgemm PRIVATE Python::Python)
set_target_properties(cgemm PROPERTIES
    PREFIX ""
    CUDA_SEPARABLE_COMPILATION ON
)
