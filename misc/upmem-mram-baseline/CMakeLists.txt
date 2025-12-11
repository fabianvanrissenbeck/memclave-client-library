cmake_minimum_required(VERSION 3.20)
project(upmem_baseline C)

set(CMAKE_C_STANDARD 17)

find_package(PkgConfig REQUIRED)
pkg_check_modules(DPU REQUIRED IMPORTED_TARGET dpu)

add_executable(upmem_baseline main.c)
target_link_libraries(upmem_baseline PUBLIC PkgConfig::DPU)
