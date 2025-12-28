cmake_minimum_required(VERSION 3.20)
project(ime-client-lib C)

set(CMAKE_C_STANDARD 17)
add_subdirectory(common)

option(ENABLE_UPMEM_BENCHMARK "build the MRAM benchmark for UPMEM and not Memclave" OFF)

if (${ENABLE_UPMEM_BENCHMARK})
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(DPU REQUIRED IMPORTED_TARGET dpu)
endif()

add_library(ime-client-lib src/vud.c src/vud.h src/vud_mem.c src/vud_mem.h src/vud_ime.c src/vud_ime.h src/vud_sk.c src/vud_sk.h src/vud_pool.c src/vud_pool.h)
target_include_directories(ime-client-lib PUBLIC src)
target_include_directories(ime-client-lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples/xchg/mbedtls/include)
target_link_libraries(ime-client-lib PUBLIC common)
target_link_libraries(ime-client-lib PUBLIC mbedcrypto)
target_compile_options(ime-client-lib PRIVATE "-O3")
target_link_libraries(ime-client-lib PUBLIC m)

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

add_executable(ime-va-example examples/VA/va.c)
target_link_libraries(ime-va-example PUBLIC ime-client-lib)

add_executable(ime-hsts-example examples/HST-S/hsts.c)
target_link_libraries(ime-hsts-example PUBLIC ime-client-lib)

add_executable(ime-hstl-example examples/HST-L/hstl.c)
target_link_libraries(ime-hstl-example PUBLIC ime-client-lib)

add_executable(ime-scanssa-example examples/SCAN-SSA/scanssa.c)
target_link_libraries(ime-scanssa-example PUBLIC ime-client-lib)

add_executable(ime-scanrss-example examples/SCAN-RSS/scanrss.c)
target_link_libraries(ime-scanrss-example PUBLIC ime-client-lib)

add_executable(ime-red-example examples/RED/red.c)
target_link_libraries(ime-red-example PUBLIC ime-client-lib)

add_executable(ime-trns-example examples/TRNS/trns.c)
target_link_libraries(ime-trns-example PUBLIC ime-client-lib)

add_executable(ime-sel-example examples/SEL/sel.c)
target_link_libraries(ime-sel-example PUBLIC ime-client-lib)

add_executable(ime-uni-example examples/UNI/uni.c)
target_link_libraries(ime-uni-example PUBLIC ime-client-lib)

add_executable(ime-bs-example examples/BS/bs.c)
target_link_libraries(ime-bs-example PUBLIC ime-client-lib)

add_executable(ime-ts-example examples/TS/ts.c)
target_link_libraries(ime-ts-example PUBLIC ime-client-lib)

add_executable(ime-spmv-example examples/SpMV/spmv.c)
target_link_libraries(ime-spmv-example PUBLIC ime-client-lib)

add_executable(ime-nw-example examples/NW/nw.c)
target_link_libraries(ime-nw-example PUBLIC ime-client-lib)

add_executable(debug examples/debug/main.c)
target_link_libraries(debug PUBLIC ime-client-lib)

add_executable(mram examples/mram/main.c)

if (${ENABLE_UPMEM_BENCHMARK})
    target_link_libraries(mram PUBLIC PkgConfig::DPU)
    target_compile_definitions(mram PRIVATE -DUSE_UPMEM=1)
else()
    target_link_libraries(mram PUBLIC ime-client-lib)
    target_compile_definitions(mram PRIVATE -DUSE_UPMEM=0)
endif()

add_executable(crypto-bench examples/crypto-bench/main.c)
target_link_libraries(crypto-bench PUBLIC ime-client-lib)

add_executable(fsl-test examples/fsl-test/main.c)
target_link_libraries(fsl-test PUBLIC ime-client-lib)
target_link_libraries(fsl-test PUBLIC mbedcrypto)
target_include_directories(fsl-test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/examples/xchg/mbedtls/include)

add_subdirectory(examples/xchg/mbedtls)
add_executable(xchg examples/xchg/main.c)
target_link_libraries(xchg PUBLIC ime-client-lib)
target_link_libraries(xchg PUBLIC mbedcrypto)
target_include_directories(xchg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/examples/xchg/mbedtls/include)
