/* vud.h - vpim userspace driver */

#ifndef ROUTER_VUD_H
#define ROUTER_VUD_H

#include <stddef.h>
#include <stdint.h>

typedef enum vud_error {
    VUD_OK,
    VUD_RANK_BUSY,
    VUD_NOT_FOUND,
    VUD_MEMORY_ERR,
    VUD_SYSTEM_ERR,
    VUD_CI_TIMEOUT,
    VUD_INVALID_RES,
    VUD_EXPECTED_FAULT,
    VUD_SK_NOT_FOUND,
    VUD_NOT_WAITING,
} vud_error;

typedef struct vud_rank {
    volatile void* base;
    int fd;
    vud_error err;
} vud_rank;

/**
 * @brief allocate a single vud rank
 * @param rank_nr number of the rank to allocate or -1 for any available rank
 */
vud_rank vud_rank_alloc(int rank_nr);

/**
 * @brief release the rank back to the os and free associated resources
 * @param rank rank to free
 */
void vud_rank_free(vud_rank* rank);

/**
 * @brief get the current mux state of one DPU line of a rank
 * @param rank rank to query
 * @returns mask where 1 bits indicate a host facing MUX
 */
uint8_t vud_rank_qry_mux(vud_rank* rank);

/**
 * @brief release control of the MUX continuing execution on the DPUs
 * @param rank rank to release MUX on
 */
void vud_rank_rel_mux(vud_rank* rank);


#endif
