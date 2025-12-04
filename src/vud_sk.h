#ifndef VUD_SK_H
#define VUD_SK_H

#include <stddef.h>
#include <stdint.h>

/**
 * @brief load an elf file and convert it into subkernel format
 *
 * This function basically wraps a few llvm-objcopy calls and as
 * such, strongly depends on its presence. GNU's objcopy does not
 * work because it cannot recognise the DPU architecture. Neither
 * does LLVM, but it treats it as being some random 32-bit architecture.
 *
 * @param path path to load elf from
 * @param sz size of the output buffer
 * @param out location to write subkernel to
 * @return size of the subkernel on success or negative value on failure
 */
long vud_sk_from_elf(const char* path, size_t sz, uint64_t* out);

/**
 * @brief encrypt the passed in subkernel under key - leave only header as AD
 * @param sk subkernel to encrypt
 * @param key key to use for encryption
 * @return 0 on success or negative value on failure
 */
int vud_enc_auth_sk(uint64_t* sk, const uint8_t* key);

#endif
