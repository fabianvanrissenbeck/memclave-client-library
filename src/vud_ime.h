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
__attribute__((deprecated))
void vud_ime_launch_sk(vud_rank* r, const char* path);

/**
 * @brief set the next subkernel (ELF file not .sk) to load
 *
 * This function sets the next subkernel to load in the rank structure.
 * Functions looking up symbol locations use this information. The
 * vud_ime_launch function also relies on this information.
 *
 * @param r rank to change the next subkernel location off
 * @param path path to a subkernel
 */
void vud_ime_load(vud_rank* r, const char* path);

/**
 * @brief load a subkernel (ELF file not .sk) on a rank of DPUs
 *
 * This function requires that a key has been installed on the rank.
 * It will load the ELF file into memory, convert it into the SK format
 * and then encrypt and tag it. Only then will it be sent to the rank.
 *
 * The file to load can be set using the vud_ime_load function.
 *
 * @param r rank to load subkernel on
 */
void vud_ime_launch(vud_rank* r);

/**
 * @brief launch on of the system subkernels
 * @param r rank to launch on
 * @param kernel chosen subkernel - note that the xchg sks call each other
 */
void vud_ime_launch_default(vud_rank* r, vud_ime_default_kernel kernel);

/**
 * @brief perform a key exchange with the rank and install a new user key
 * @param r rank to install key on
 * @param key 256-bit key to install
 * @param common_pk public key to expect of all DPUs - may be NULL
 * @param pk public key to expect from each DPU - preferred if not NULL
 */
void vud_ime_install_key(vud_rank* r, const uint8_t key[32], const uint64_t common_pk[32], const uint64_t pk[64][32]);

/**
 * @brief deploy multiple subkernels to a rank of DPUs and launch the first one
 * @param r rank to deploy subkernels on
 * @param n number of subkernels
 * @param paths array of paths to subkernels
 * @param addrs addresses to deploy subkernels to - should not alias each other
 */
__attribute__((deprecated))
void vud_ime_launch_sk_ext(vud_rank* r, size_t n, const char** paths, const uint64_t* addrs);

/**
 * @brief wait until the whole rank has exposed the MUX to the guest system
 * @param r pointer to the concrete rank
 */
void vud_ime_wait(vud_rank* r);

#endif
