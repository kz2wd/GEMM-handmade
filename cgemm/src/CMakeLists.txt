file(GLOB_RECURSE ALL_SRC_FILES "*.c" "*.cu")

add_library(cgemm MODULE
    ${ALL_SRC_FILES}
)

target_include_directories(cgemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_source_files_properties(
    cgemm_kernel.c
    PROPERTIES
    COMPILE_OPTIONS "-fopt-info-optimized;-fopt-info-missed"
)

add_executable(standalone_runner ${ALL_SRC_FILES})

target_link_libraries(standalone_runner PRIVATE Python::Python)

target_compile_options(standalone_runner PRIVATE
    -mavx2
    -mfma
    -march=native
    -O3
    -g
    -pg
)
target_link_options(standalone_runner PRIVATE
    -pg
)
