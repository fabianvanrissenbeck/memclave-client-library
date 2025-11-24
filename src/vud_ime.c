#include "vud_ime.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>

#define IME_MSG_BUFFER 0x03fc0000
#define IME_LOAD_BUFFER (IME_MSG_BUFFER + sizeof(ime_mram_msg))

typedef enum ime_mram_msg_type {
    IME_MRAM_MSG_NOP,
    IME_MRAM_MSG_WAITING,
    IME_MRAM_MSG_PING,
    IME_MRAM_MSG_PONG,
    IME_MRAM_MSG_READ_WRAM,
    IME_MRAM_MSG_WRITE_WRAM,
    IME_MRAM_MSG_LOAD_SK,
} ime_mram_msg_type;

typedef struct ime_mram_msg {
    uint32_t type;
    union {
        struct {
            uint32_t addr;
            uint32_t value;
        } wram;
        struct {
            uint32_t ptr;
        } load;
        struct {
            uint32_t pad[3];
        };
    };
} ime_mram_msg;

_Static_assert(__builtin_offsetof(ime_mram_msg, wram.addr) == 4, "incorrect alignment");
_Static_assert(__builtin_offsetof(ime_mram_msg, wram.value) == 8, "incorrect alignment");

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
        ptr[i] = (uint64_t*) &buf[i];
    }

    vud_simple_gather(r, 2, IME_MSG_BUFFER, &ptr);
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
        .load.ptr = sk_addr_table[kernel],
    };

    const uint64_t* msg_ptr = (const uint64_t*) &msg;
    uint64_t arr[2] = { msg_ptr[0], msg_ptr[1] };

    vud_broadcast_transfer(r, 2, &arr, IME_MSG_BUFFER);

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
        .load.ptr = IME_LOAD_BUFFER,
    };

    const uint64_t* msg_ptr = (const uint64_t*) &msg;
    uint64_t arr[2] = { msg_ptr[0], msg_ptr[1] };

    vud_broadcast_transfer(r, 2, &arr, IME_MSG_BUFFER);

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
        .load.ptr = addrs[0],
    };

    const uint64_t* msg_ptr = (const uint64_t*) &msg;
    uint64_t arr[2] = { msg_ptr[0], msg_ptr[1] };

    vud_broadcast_transfer(r, 2, &arr, IME_MSG_BUFFER);

    if (r->err) { return; }
    vud_rank_rel_mux(r);
}

void vud_ime_wait(vud_rank* r) {
    if (r->err) { return; }
    wait_for_fault(r);
}