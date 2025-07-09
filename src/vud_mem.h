#ifndef ROUTER_VUD_MEM_H
#define ROUTER_VUD_MEM_H

#include "vud.h"

#include <stdint.h>

typedef uint32_t vud_mram_addr;
typedef uint32_t vud_mram_size;

/**
 * @brief transfer the same buffer to all DPUs of one rank
 * @param r rank to transfer buffer to
 * @param sz amount of words (8-bytes each) to transfer
 * @param src buffer holding data to transfer
 * @param tgt mram address to copy buffer to
 */
void vud_broadcast_transfer(vud_rank* r, vud_mram_size sz, const uint64_t (*src)[sz], vud_mram_addr tgt);

/**
 * @brief transfer 64 possibly different buffers to the DPUs of one rank
 * @param r rank to transfer to
 * @param sz amount of words (8-bytes each) to transfer
 * @param src array of pointers to the 64 buffers
 * @param tgt mram address to copy buffers to (same for all DPUs)
 */
void vud_simple_transfer(vud_rank* r, vud_mram_size sz, const uint64_t* (*src)[64], vud_mram_addr tgt);

/**
 * @brief transfer words from all DPUs to the host
 * @param r rank to transfer bytes from
 * @param sz amount of words (8-bytes each) to transfer
 * @param src Address to read from, the same for all DPUs.
 * @param tgt Buffer to write words to.
 */
void vud_simple_gather(vud_rank* r, vud_mram_size sz, vud_mram_addr src, uint64_t* (*tgt)[64]);

#endif
