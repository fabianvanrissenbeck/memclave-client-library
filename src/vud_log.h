// src/vud_log.h
#ifndef VUD_LOG_H
#define VUD_LOG_H

#include "vud.h"
#include "vud_mem.h"

#include <stdint.h>

/// total MRAM per DPU
#define MRAM_SIZE_BYTES     (64u << 20)
/// we reserve 64 B at the very top
#define SK_LOG_SIZE_BYTES   64
#define SK_LOG_OFFSET       (MRAM_SIZE_BYTES - SK_LOG_SIZE_BYTES)
/// number of 8‑byte slots
#define SK_LOG_MAX_ENTRIES  (SK_LOG_SIZE_BYTES / sizeof(uint64_t))

/// Read back SK_LOG_MAX_ENTRIES words from each DPU’s MRAM
/// into logs[dpu_id][slot_id]. Returns r->err.
static inline int vud_log_read(vud_rank *r,
                               int nb_dpus,
                               uint64_t logs[][SK_LOG_MAX_ENTRIES]) {
    if (nb_dpus > 64) 
	    nb_dpus = 64;

    uint8_t *dsts[64];
    for (int d = 0; d < nb_dpus; ++d) {
        dsts[d] = (uint8_t *)logs[d];
    }
    vud_simple_gather(r,
                      SK_LOG_MAX_ENTRIES,  // number of 8‑byte words
                      SK_LOG_OFFSET,       // byte offset in MRAM
                      dsts);
    return r->err;
}

#endif // VUD_LOG_H

