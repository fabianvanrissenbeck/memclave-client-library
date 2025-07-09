#ifndef VUD_IME_H
#define VUD_IME_H

#include "vud.h"
#include "vud_mem.h"

/**
 * @brief launch a subkernel on a rank of DPUs
 * @param r pointer to the concrete rank
 * @param path path to the subkernel
 */
void vud_ime_launch_sk(vud_rank* r, const char* path);

/**
 * @brief wait until the whole rank has exposed the MUX to the guest system
 * @param r pointer to the concrete rank
 */
void vud_ime_wait(vud_rank* r);

#endif
