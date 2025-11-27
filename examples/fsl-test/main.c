#include <vud.h>
#include <vud_mem.h>
#include <vud_ime.h>

#include "mbedtls/bignum.h"
#include "mbedtls/sha256.h"
#include "mbedtls/dhm.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>

static uint8_t mbedtls_dhm_prime[] = MBEDTLS_DHM_RFC3526_MODP_2048_P_BIN;
static uint8_t mbedtls_dhm_group[] = MBEDTLS_DHM_RFC3526_MODP_2048_G_BIN;

#define IME_DPU_CNTR 0x0
#define IME_CLIENT_PUBKEY 0x110
#define IME_DPU_PUBKEY 0x10

void buf_to_stdout(size_t sz, const uint64_t* buf) {
    FILE* p = popen("xxd -e -g 8", "w");
    assert(p != NULL);

    fwrite(buf, 1, sz * sizeof(buf[0]), p);
    pclose(p);
}

int main(void) {
    vud_rank r = vud_rank_alloc(VUD_ALLOC_ANY);

    if (r.err) {
        printf("cannot allocate rank: %s\n", vud_error_str(r.err));
        return -1;
    }

    vud_ime_install_key(&r, NULL, NULL, NULL);

    if (r.err) {
        printf("cannot exchange keys: %s\n", vud_error_str(r.err));
        return -1;
    }

    puts("Done.");
    vud_rank_free(&r);

    return 0;
}