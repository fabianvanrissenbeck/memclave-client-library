#ifndef COMMON_AES_H
#define COMMON_AES_H

#include <stdint.h>

/**
 * @brief single block used in AES128 - Can be either a key or a block to encrypt
 */
typedef struct aes_block {
    union {
        uint8_t mat[16];
        uint32_t columns[4];
    };
} aes_block;

_Static_assert(sizeof(aes_block) == 16, "aes block needs to be exactly 128-bits large");

#endif
