/* vud.h - vpim userspace driver */

#ifndef ROUTER_VUD_H
#define ROUTER_VUD_H

#include <stddef.h>
#include <stdint.h>

#define VUD_ALLOC_ANY -1

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
    uint8_t key[32];
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

/**
 * @brief get an error string corresponding to the passed in value
 * @param err error value to convert
 * @return statically allocated string representation
 */
static inline const char* vud_error_str(vud_error err) {
    static const char* table[VUD_NOT_WAITING + 1] = {
        [VUD_OK] = "success",
        [VUD_RANK_BUSY] = "rank is busy",
        [VUD_NOT_FOUND] = "rank not found",
        [VUD_MEMORY_ERR] = "memory error (mmap)",
        [VUD_SYSTEM_ERR] = "system error (open)",
        [VUD_CI_TIMEOUT] = "ci access timed out",
        [VUD_INVALID_RES] = "invalid VCI response",
        [VUD_EXPECTED_FAULT] = "DPUs are not all in fault",
        [VUD_SK_NOT_FOUND] = "could not find requested subkernel",
        [VUD_NOT_WAITING] = "DPU is not waiting for guest",
    };

    if (err >= VUD_OK && err <= VUD_NOT_WAITING) {
        return table[err];
    }

    return "unknown error";
}


#endif
