#include "vud_ime.h"

#include "mbedtls/chachapoly.h"
#include "mbedtls/bignum.h"
#include "mbedtls/sha256.h"
#include "mbedtls/dhm.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>

#define IME_MSG_BUFFER 0x03fc0000
#define IME_LOAD_BUFFER (IME_MSG_BUFFER + sizeof(ime_mram_msg))
#define IME_DPU_CNTR 0x0
#define IME_CLIENT_PUBKEY 0x110
#define IME_DPU_PUBKEY 0x10
#define IME_KEY_IN 0x210

static uint8_t mbedtls_dhm_prime[] = MBEDTLS_DHM_RFC3526_MODP_2048_P_BIN;
static uint8_t mbedtls_dhm_group[] = MBEDTLS_DHM_RFC3526_MODP_2048_G_BIN;

typedef enum ime_mram_msg_type {
    IME_MRAM_MSG_NOP,
    IME_MRAM_MSG_WAITING,
    IME_MRAM_MSG_PING,
    IME_MRAM_MSG_PONG,
    IME_MRAM_MSG_LOAD_SK,
} ime_mram_msg_type;

typedef enum ime_load_sk_flags {
    IME_LOAD_SK_USER_KEY = 1 << 0,
} ime_load_sk_flags;

typedef union ime_mram_msg {
    struct {
        uint16_t type;
        uint16_t flags;
        uint32_t ptr;
    };
    uint64_t raw;
} ime_mram_msg;

static void wait_for_fault(vud_rank* r) {
    uint8_t n_faulted = 0;
    uint64_t timeout = 0;

    do {
        n_faulted = vud_rank_qry_mux(r);

        if (r->err) {
            printf("Cannot query MUX of rank.\n");
            return;
        }

        usleep(timeout);

        /* exponential back-off until a 25ms wait period */
        if (timeout == 0) {
            timeout = 1;
        } else if (timeout < 25000) {
            timeout = timeout << 1;
        }
    } while (n_faulted != 64);
}

static uint8_t* load_file(const char* path, size_t* out_sz) {
    FILE* fp = fopen(path, "rb");
    if (fp == NULL) { return NULL; }

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    uint8_t* res = malloc(size);
    assert(res != NULL);

    assert(fread(res, 1, size, fp) == size);

    if (out_sz) {
        *out_sz = size;
    }

    fclose(fp);
    return res;
}

static void poll_ime_msg(vud_rank* r, ime_mram_msg buf[64]) {
    uint64_t* ptr[64];

    for (int i = 0; i < 64; i++) {
        ptr[i] = &buf[i].raw;
    }

    vud_simple_gather(r, 1, IME_MSG_BUFFER, &ptr);
}

static void vud_check_launchable(vud_rank* r) {
    if (r->err) { return; }

    /* all DPUs must be in fault before we can transmit any data to MRAM */
    if (vud_rank_qry_mux(r) != 64) {
        r->err = VUD_EXPECTED_FAULT;
    }

    if (r->err) { return; }

    /* make sure that all DPUs requested an MRAM message (makes sure the message subkernel is running) */
    ime_mram_msg msg_buf[64];
    poll_ime_msg(r, msg_buf);

    for (int i = 0; i < 64; ++i) {
        if (msg_buf[i].type != IME_MRAM_MSG_WAITING) {
            r->err = VUD_NOT_WAITING;
            return;
        }
    }
}

void vud_ime_launch_default(vud_rank* r, vud_ime_default_kernel kernel) {
    static uint64_t sk_addr_table[] = {
        [VUD_IME_SK_MSG] = 0x3f00000,
        [VUD_IME_SK_XCHG_1] = 0x3f20000,
        [VUD_IME_SK_XCHG_2] = 0x3f40000,
        [VUD_IME_SK_XCHG_3] = 0x3f60000,
    };

    vud_check_launchable(r);
    if (r->err) { return; }

    ime_mram_msg msg = {
        .type = IME_MRAM_MSG_LOAD_SK,
        .ptr = sk_addr_table[kernel],
        .flags = 0
    };

    const uint64_t arr[1] = { msg.raw };
    vud_broadcast_transfer(r, 1, &arr, IME_MSG_BUFFER);

    if (r->err) { return; }
    vud_rank_rel_mux(r);
}

void vud_ime_launch_sk(vud_rank* r, const char* path) {
    vud_check_launchable(r);
    if (r->err) { return; }

    size_t sk_size;
    uint64_t* sk = (uint64_t*) load_file(path, &sk_size);

    if (sk == NULL) {
        r->err = VUD_SK_NOT_FOUND;
        return;
    }

    vud_broadcast_transfer(r, sk_size / 8, (const uint64_t (*)[]) sk, IME_LOAD_BUFFER);

    ime_mram_msg msg = {
        .type = IME_MRAM_MSG_LOAD_SK,
        .ptr = IME_LOAD_BUFFER,
        .flags = 0
    };

    uint64_t arr[1] = { msg.raw };
    vud_broadcast_transfer(r, 1, &arr, IME_MSG_BUFFER);

    if (r->err) { return; }
    vud_rank_rel_mux(r);
}

void vud_ime_launch_sk_ext(vud_rank* r, size_t n, const char** paths, const uint64_t* addrs) {
    if (r->err) { return; }

    /* all DPUs must be in fault before we can transmit any data to MRAM */
    if (vud_rank_qry_mux(r) != 64) {
        r->err = VUD_EXPECTED_FAULT;
    }

    if (r->err) { return; }

    /* make sure that all DPUs requested an MRAM message (makes sure the message subkernel is running) */
    ime_mram_msg msg_buf[64];
    poll_ime_msg(r, msg_buf);

    for (int i = 0; i < 64; ++i) {
        if (msg_buf[i].type != IME_MRAM_MSG_WAITING) {
            r->err = VUD_NOT_WAITING;
            return;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        size_t sk_size;
        uint64_t* sk = (uint64_t*) load_file(paths[i], &sk_size);

        if (sk == NULL) {
            r->err = VUD_SK_NOT_FOUND;
            return;
        }

        vud_broadcast_transfer(r, sk_size / 8, (const uint64_t (*)[]) sk, addrs[i]);
    }

    ime_mram_msg msg = {
        .type = IME_MRAM_MSG_LOAD_SK,
        .ptr = addrs[0],
        .flags = 0
    };

    const uint64_t* msg_ptr = (const uint64_t*) &msg;
    uint64_t arr[2] = { msg_ptr[0], msg_ptr[1] };

    vud_broadcast_transfer(r, 1, &arr, IME_MSG_BUFFER);

    if (r->err) { return; }
    vud_rank_rel_mux(r);
}

void vud_ime_wait(vud_rank* r) {
    if (r->err) { return; }
    wait_for_fault(r);
}

void buf_to_stdout(size_t sz, const uint64_t* buf) {
    FILE* p = popen("xxd -e -g 8", "w");
    assert(p != NULL);

    fwrite(buf, 1, sz * sizeof(buf[0]), p);
    pclose(p);
}

void vud_ime_install_key(vud_rank* r, const uint8_t key[32], const uint64_t common_pk[32], const uint64_t specific_pk[64][32]) {
    if (r->err) { return; }

    puts("Waiting for DPU");
    vud_ime_wait(r);
    puts("Done waiting for DPU");

    vud_mram_addr output_addr = (64 << 20) - 64;
    vud_broadcast_transfer(r, 8, &(const uint64_t[8]) { 0 }, output_addr);

    vud_ime_launch_default(r, VUD_IME_SK_XCHG_1);
    puts("Launched XCHG1");

    vud_ime_wait(r);
    puts("Received Pubkey Request");

    uint32_t sk_raw[8];
    FILE* fp_rand = fopen("/dev/urandom", "rb");

    assert(fp_rand != NULL);
    assert(fread(sk_raw, 1, sizeof(sk_raw), fp_rand) == sizeof(sk_raw));

    fclose(fp_rand);

    mbedtls_mpi pk;
    mbedtls_mpi sk;
    mbedtls_mpi p;
    mbedtls_mpi g;
    mbedtls_mpi dpu_pk[64];
    mbedtls_mpi shared[64];

    mbedtls_mpi_init(&pk);
    mbedtls_mpi_init(&sk);
    mbedtls_mpi_init(&p);
    mbedtls_mpi_init(&g);

    for (int i = 0; i < 64; i++) {
        mbedtls_mpi_init(&dpu_pk[i]);
        mbedtls_mpi_init(&shared[i]);
    }

    mbedtls_mpi_read_binary_le(&sk, sk_raw, sizeof(sk_raw));
    mbedtls_mpi_read_binary(&p, mbedtls_dhm_prime, sizeof(mbedtls_dhm_prime));
    mbedtls_mpi_read_binary(&g, mbedtls_dhm_group, sizeof(mbedtls_dhm_group));
    mbedtls_mpi_exp_mod(&pk, &g, &sk, &p, NULL);

    uint64_t dpu_pub_raw[64][32];
    uint64_t* dpu_pub_raw_ptr[64];
    uint64_t dpu_ctr[64][2];
    uint64_t* dpu_ctr_ptr[64];

    for (int i = 0; i < 64; ++i) {
        dpu_pub_raw_ptr[i] = dpu_pub_raw[i];
        dpu_ctr_ptr[i] = dpu_ctr[i];
    }

    vud_simple_gather(r, 32, IME_DPU_PUBKEY, &dpu_pub_raw_ptr);
    vud_simple_gather(r, 2, IME_DPU_CNTR, &dpu_ctr_ptr);

    if (r->err) { return; }

    if (specific_pk || common_pk) {
        for (int i = 0; i < 64; ++i) {
            const uint64_t* good_pk = specific_pk ? specific_pk[i] : common_pk;

            if (memcmp(dpu_pub_raw[i], good_pk, sizeof(dpu_pub_raw[i])) != 0) {
                puts("DPU offered unknown public key");
                r->err = VUD_KEY_XCHG;
            }
        }
    }

    if (r->err) { return; }

    for (int i = 0; i < 64; ++i) {
        mbedtls_mpi_read_binary_le(&dpu_pk[i], (const uint8_t*) &dpu_pub_raw[i][0], sizeof(dpu_pub_raw[i]));
        mbedtls_mpi_exp_mod(&shared[i], &dpu_pk[i], &sk, &p, NULL);
    }

    uint8_t shared_sec[64][32];

    for (int i = 0; i < 64; ++i) {
        mbedtls_sha256_context sha_ctx;

        mbedtls_sha256_init(&sha_ctx);
        mbedtls_sha256_starts(&sha_ctx, 0);
        mbedtls_sha256_update(&sha_ctx, (const uint8_t*) shared[i].private_p, sizeof(uint64_t) * 32);
        mbedtls_sha256_update(&sha_ctx, (const uint8_t*) dpu_ctr[i], 16);
        mbedtls_sha256_finish(&sha_ctx, shared_sec[i]);

        if (i == 0) {
            buf_to_stdout(sizeof(shared_sec[i]) / sizeof(uint64_t), (const uint64_t*) &shared_sec[i][0]);
        }
    }

    uint8_t key_in[64][32 + 16];
    const uint64_t* key_in_ptr[64];

    for (int i = 0; i < 64; ++i) {
        mbedtls_chachapoly_context ctx;

        mbedtls_chachapoly_init(&ctx);
        mbedtls_chachapoly_setkey(&ctx, shared_sec[i]);
        mbedtls_chachapoly_encrypt_and_tag(&ctx, 32, (uint8_t[12]) { 0 }, NULL, 0, key, &key_in[i][0], &key_in[i][32]);
        mbedtls_chachapoly_free(&ctx);

        key_in_ptr[i] = (const uint64_t*) &key_in[i][0];
    }

    vud_simple_transfer(r, 6, &key_in_ptr, IME_KEY_IN);
    vud_broadcast_transfer(r, 32, pk.private_p, IME_CLIENT_PUBKEY);
    vud_rank_rel_mux(r);

    puts("Answered PubKey Request");
    vud_ime_wait(r);

    uint64_t data[64][8];
    uint64_t* data_ptr[64];

    for (int i = 0; i < 64; ++i) { data_ptr[i] = data[i]; }

    vud_simple_gather(r, 8, output_addr, &data_ptr);

    if (r->err) { return; }

    for (int j = 0; j < 64; ++j) {
        if (memcmp(&data[j][0], key, 32) != 0) {
            printf("========== DPU %02o Failed ==========\n", j);
            buf_to_stdout(8, &data[j][0]);
            r->err = VUD_KEY_XCHG;
        }
    }
}
