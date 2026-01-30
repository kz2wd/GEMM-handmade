file(GLOB_RECURSE ALL_SRC_FILES "*.c" "*.cu")

add_library(cgemm MODULE
    ${ALL_SRC_FILES}
)

target_include_directories(cgemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

