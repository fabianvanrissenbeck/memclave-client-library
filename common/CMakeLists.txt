cmake_minimum_required(VERSION 3.13)
project(common C)

set(CMAKE_C_STANDARD 11)

add_library(common INTERFACE)
target_include_directories(common INTERFACE include/)