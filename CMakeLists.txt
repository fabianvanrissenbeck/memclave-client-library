cmake_minimum_required(VERSION 3.20)
project(ime-client-lib C)

set(CMAKE_C_STANDARD 17)
add_subdirectory(common)

add_library(ime-client-lib src/vud.c src/vud.h src/vud_mem.c src/vud_mem.h src/vud_ime.c src/vud_ime.h)
target_include_directories(ime-client-lib PUBLIC src)
target_link_libraries(ime-client-lib PUBLIC common)

add_executable(ime-add-example examples/add.c)
target_link_libraries(ime-add-example PUBLIC ime-client-lib)

add_executable(ime-mbench-example examples/mbench/mbench.c)
target_link_libraries(ime-mbench-example PUBLIC ime-client-lib)

add_executable(ime-gemv-example examples/gemv/gemv.c)
target_link_libraries(ime-gemv-example PUBLIC ime-client-lib)

add_executable(ime-mlp-example examples/MLP/mlp.c)
target_link_libraries(ime-mlp-example PUBLIC ime-client-lib)

add_executable(multi-tasklet examples/multi-tasklet.c)
target_link_libraries(multi-tasklet PUBLIC ime-client-lib)

add_executable(ime-bfs-example examples/BFS/bfs.c)
target_link_libraries(ime-bfs-example PUBLIC ime-client-lib)

add_executable(debug examples/debug/main.c)
target_link_libraries(debug PUBLIC ime-client-lib)

