cmake_minimum_required(VERSION 3.28)
project(cgemm)

# set(CMAKE_BUILD_TYPE Debug)

find_package(Python REQUIRED COMPONENTS Development)
add_library(cgemm MODULE
    src/cgemm.c
    src/cgemm_naive.c
    src/cgemm_args.c
)

target_link_libraries(cgemm PRIVATE Python::Python)
set_target_properties(cgemm PROPERTIES PREFIX "")