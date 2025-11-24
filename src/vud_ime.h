#ifndef VUD_IME_H
#define VUD_IME_H

#include "vud.h"
#include "vud_mem.h"

typedef enum vud_ime_default_kernel {
    VUD_IME_SK_MSG = 1,
    VUD_IME_SK_XCHG_1,
    VUD_IME_SK_XCHG_2,
    VUD_IME_SK_XCHG_3,
} vud_ime_default_kernel;

/**
 * @brief launch a subkernel on a rank of DPUs
 * @param r pointer to the concrete rank
 * @param path path to the subkernel
 */
void vud_ime_launch_sk(vud_rank* r, const char* path);

void vud_ime_launch_default(vud_rank* r, vud_ime_default_kernel kernel);

/**
 * @brief deploy multiple subkernels to a rank of DPUs and launch the first one
 * @param r rank to deploy subkernels on
 * @param n number of subkernels
 * @param paths array of paths to subkernels
 * @param addrs addresses to deploy subkernels to - should not alias each other
 */
void vud_ime_launch_sk_ext(vud_rank* r, size_t n, const char** paths, const uint64_t* addrs);

/**
 * @brief wait until the whole rank has exposed the MUX to the guest system
 * @param r pointer to the concrete rank
 */
void vud_ime_wait(vud_rank* r);

#endif
