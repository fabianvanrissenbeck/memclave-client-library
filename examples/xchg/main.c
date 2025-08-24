#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "mbedtls/bignum.h"
#include "mbedtls/sha256.h"
#include "mbedtls/dhm.h"
#include "vud_ime.h"

#define IME_CLIENT_PUBKEY 0x110
#define IME_DPU_PUBKEY 0x10
#define VUD_IME_XCHG_1 0x03f20000
#define VUD_IME_XCHG_2 0x03f40000
#define VUD_IME_XCHG_3 0x03f60000

static uint8_t mbedtls_dhm_prime[] = MBEDTLS_DHM_RFC3526_MODP_2048_P_BIN;
static uint8_t mbedtls_dhm_group[] = MBEDTLS_DHM_RFC3526_MODP_2048_G_BIN;

void buf_to_stdout(size_t sz, const uint64_t* buf) {
    FILE* p = popen("xxd -e -g 8", "w");
    assert(p != NULL);

    fwrite(buf, 1, sz * sizeof(buf[0]), p);
    pclose(p);
}

int main(int argc, char** argv) {
#if 0
    if (argc != 2) {
        puts("debug - Program to run a single subkernel and print the debug output.");
        puts("Usage: ./debug <subkernel>");

        return 1;
    }
#endif

    vud_rank r = vud_rank_alloc(10);

    if (r.err) {
        puts("Cannot allocate rank.");
        return 1;
    }

    vud_ime_wait(&r);

    if (r.err) {
        puts("Cannot wait for rank.");
        return 1;
    }

    vud_mram_addr output_addr = (64 << 20) - 64;
    vud_broadcast_transfer(&r, 8, &(uint64_t[8]) { 0 }, output_addr);

    if (r.err) {
        puts("Cannot zero out prior results.");
        return 1;
    }

    vud_ime_launch_sk_ext(
        &r, 3,
        (const char*[]) { "../xchg1.sk", "../xchg2.sk", "../xchg3.sk" },
        (const uint64_t[]) { VUD_IME_XCHG_1, VUD_IME_XCHG_2, VUD_IME_XCHG_3 }
    );

    if (r.err) {
        printf("Cannot launch subkernel: vud error %d\n", r.err);
        return 1;
    }

    vud_ime_wait(&r);

    if (r.err) {
        puts("cannot wait for client pubkey request");
        return 1;
    }

#if 0
    uint32_t sk_raw[8] = {
        0xe94dc837, 0x29042514, 0x87826031, 0xcdacd9b0,
        0x27da09e9, 0x16a9af3c, 0xbc800e02, 0x74fbb2a2  // 0x74fbb2a1,
    };
#else
    uint32_t sk_raw[8];
    FILE* fp_rand = fopen("/dev/urandom", "rb");

    assert(fp_rand != NULL);
    assert(fread(sk_raw, 1, sizeof(sk_raw), fp_rand) == sizeof(sk_raw));

    fclose(fp_rand);
#endif

    mbedtls_mpi pk;
    mbedtls_mpi sk;
    mbedtls_mpi p;
    mbedtls_mpi g;
    mbedtls_mpi dpu_pk;
    mbedtls_mpi shared;

    mbedtls_mpi_init(&pk);
    mbedtls_mpi_init(&sk);
    mbedtls_mpi_init(&p);
    mbedtls_mpi_init(&g);
    mbedtls_mpi_init(&shared);

    mbedtls_mpi_read_binary_le(&sk, sk_raw, sizeof(sk_raw));
    mbedtls_mpi_read_binary(&p, mbedtls_dhm_prime, sizeof(mbedtls_dhm_prime));
    mbedtls_mpi_read_binary(&g, mbedtls_dhm_group, sizeof(mbedtls_dhm_group));
    mbedtls_mpi_exp_mod(&pk, &g, &sk, &p, NULL);

    uint64_t dpu_pub_raw[64][32];
    uint64_t* dpu_pub_raw_ptr[64];

    for (int i = 0; i < 64; ++i) { dpu_pub_raw_ptr[i] = dpu_pub_raw[i]; }

    vud_simple_gather(&r, 32, IME_DPU_PUBKEY, &dpu_pub_raw_ptr);

    if (r.err) {
        puts("cannot fetch DPU public key");
        return 1;
    }

    mbedtls_mpi_read_binary_le(&dpu_pk, (const uint8_t*) &dpu_pub_raw[0][0], sizeof(dpu_pub_raw[0]));
    mbedtls_mpi_exp_mod(&shared, &dpu_pk, &sk, &p, NULL);

    mbedtls_sha256_context sha_ctx;
    uint8_t key[32];

    mbedtls_sha256_init(&sha_ctx);
    mbedtls_sha256_starts(&sha_ctx, 0);
    mbedtls_sha256_update(&sha_ctx, (const uint8_t*) shared.private_p, sizeof(uint64_t) * 32);
    mbedtls_sha256_update(&sha_ctx, (uint8_t[16]) { 0 }, 16);
    mbedtls_sha256_finish(&sha_ctx, key);

    buf_to_stdout(32 / 8, key);

    vud_broadcast_transfer(&r, 32, pk.private_p, IME_CLIENT_PUBKEY);

    if (r.err) {
        puts("cannot share public key");
        return 1;
    }

    vud_rank_rel_mux(&r);

    if (r.err) {
        puts("cannot answer client pubkey request");
    }

    uint64_t data[64][8];
    uint64_t* data_ptr[64];

    for (int i = 0; i < 64; ++i) { data_ptr[i] = data[i]; }

    vud_ime_wait(&r);

    if (r.err) {
        puts("could not wait for subkernel completion");
        return 1;
    }

    vud_simple_gather(&r, 8, output_addr, &data_ptr);

    for (int j = 0; j < 64; ++j) {
        if (memcmp(&data[j][0], key, 32) != 0) {
            printf("========== DPU %02o Failed ==========\n", j);
            buf_to_stdout(8, &data[j][0]);
        }
    }

    vud_rank_free(&r);
    return 0;
}