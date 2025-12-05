#include "vud_sk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mbedtls/chachapoly.h>

/**
 * @brief load a random IV from /dev/urandom where the lowest bit is 0 so it
 * cannot collide with the DPUs IV
 * @param out location to write generated IV to
 * @return 0 on success or negativ value on failure
 */
static int gen_good_iv(uint8_t out[12]) {
    FILE* fp = fopen("/dev/urandom", "rb");

    if (fp == NULL) { return -1; }
    if (fread(out, 1, 12, fp) != 12) { return -1; }

    fclose(fp);
    out[0] &= 0xFE;

    return 0;
}

long vud_sk_from_elf(const char* path, size_t sz, uint64_t* out) {
    char cmd_text[240];
    char cmd_data[240];

    snprintf(cmd_text, sizeof(cmd_text), "llvm-objcopy -O binary --only-section .text %s -", path);
    snprintf(cmd_data, sizeof(cmd_data), "llvm-objcopy -O binary --only-section .data %s -", path);

    FILE* sec_text = popen(cmd_text, "r");
    FILE* sec_data = popen(cmd_data, "r");

    if (sec_text == NULL || sec_data == NULL) { goto failure; }

    // neither .text nor .data are larger than 64 KiB - its easier to just
    // overallocate a bit than to use a dynamic array

    uint64_t* buf_text = calloc(64 << 10, 1);
    uint64_t* buf_data = calloc(64 << 10, 1);

    if (buf_text == NULL || buf_data == NULL) { goto failure; }

    size_t n_text = fread(buf_text, 1, 64 << 10, sec_text);
    size_t n_data = fread(buf_data, 1, 64 << 10, sec_data);

    if (n_text % 2048 != 0) {
        n_text += 2048 - (n_text % 2048);
    }

    if (n_data % 2048 != 0) {
        n_data += 2048 - (n_data % 2048);
    }

    if (pclose(sec_text) != 0) {
        sec_text = NULL;
        goto failure;
    }

    sec_text = NULL;

    if (pclose(sec_data) != 0) {
        sec_data = NULL;
        goto failure;
    }

    sec_data = NULL;

    uint32_t size_aad = 64;
    uint32_t size = 64 + n_text + n_data;

    uint64_t header[8] = {
        0x00000000A5A5A5A5,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
        size_aad | (uint64_t)(size) << 32,
        (n_text / 2048) | (uint64_t)(n_data / 2048) << 32,
    };

    if (sz < sizeof(header) + n_text + n_data) {
        goto failure;
    }

    memcpy(out, header, sizeof(header));
    memcpy(&out[8], buf_text, n_text);
    memcpy(&out[8 + n_text / sizeof(uint64_t)], buf_data, n_data);

    free(buf_text);
    free(buf_data);

    buf_text = NULL;
    buf_data = NULL;

    return (long)(sizeof(header) + n_text + n_data);

failure:
    if (sec_text) { pclose(sec_text); }
    if (sec_data) { pclose(sec_data); }

    free(buf_text);
    free(buf_data);

    return -1;
}

int vud_enc_auth_sk(uint64_t* sk, const uint8_t* key) {
    mbedtls_chachapoly_context ctx_buf;
    mbedtls_chachapoly_context* ctx = NULL;
    uint8_t iv[12];
    uint8_t tag[16];

    uint32_t size = sk[4] >> 32;
    uint32_t size_aad = sk[4] & 0xffffffff;

    const uint8_t* sk_aad = (uint8_t*) sk;
    uint8_t* sk_crypt = (uint8_t*) &sk[size_aad / sizeof(uint64_t)];

    mbedtls_chachapoly_init(&ctx_buf);
    ctx = &ctx_buf;

    if (mbedtls_chachapoly_setkey(ctx, key) != 0) { goto failure; }
    if (gen_good_iv(iv) < 0) { goto failure; }

    if (mbedtls_chachapoly_encrypt_and_tag(ctx, size - size_aad, iv, sk_aad, size_aad, sk_crypt, sk_crypt, tag) != 0) {
        goto failure;
    }

    uint8_t* sk_tag = &((uint8_t*) sk)[4];
    memcpy(sk_tag, tag, 16);

    uint8_t* sk_iv = &((uint8_t*) sk)[20];
    memcpy(sk_iv, iv, 12);

    mbedtls_chachapoly_free(ctx);
    return 0;

failure:
    mbedtls_chachapoly_free(ctx);
    return -1;
}
