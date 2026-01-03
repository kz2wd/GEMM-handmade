cmake_minimum_required(VERSION 3.31)

find_package(Python REQUIRED COMPONENTS Development)
add_library(cgemm MODULE
    src/cgemm.c
)

target_link_libraries(cgemm PRIVATE Python::Python)
set_target_properties(cgemm PROPERTIES PREFIX "")