#include <stdio.h>
#include <unistd.h>
#include <assert.h>

#include "vud.h"
#include "vud_ime.h"

#define BLOCK_SIZE 24
#define SUBKERNEL "../chacha-bench"

static void random_key(uint8_t key[32]) {
    FILE* fp = fopen("/dev/urandom", "rb");

    assert(fp != NULL);
    assert(fread(key, 1, 32, fp) == 32);

    fclose(fp);
}

static void fetch_results_from(vud_rank* r) {
    uint64_t out_time[64];
    uint64_t* out_time_ptr[64];

    for (int i = 0; i < 64; ++i) {
        out_time_ptr[i] = &out_time[i];
    }

    vud_gather_from(r, 1, "out_time_enc", &out_time_ptr);

    if (r->err) {
        return;
    }

    for (int i = 0; i < 64; ++i) {
        printf("%u,%zu,enc,%02o\n", 1 << BLOCK_SIZE, out_time[i], i);
    }

    vud_gather_from(r, 1, "out_time_dec", &out_time_ptr);

    if (r->err) {
        return;
    }

    for (int i = 0; i < 64; ++i) {
        printf("%u,%zu,dec,%02o\n", 1 << BLOCK_SIZE, out_time[i], i);
    }
}

int main(int argc, char** argv) {
    uint8_t key[32];
    vud_rank rank = vud_rank_alloc(VUD_ALLOC_ANY);

    vud_ime_wait(&rank);
    vud_ime_load(&rank, SUBKERNEL);

    if (rank.err) {
        puts("cannot load subkernel");
        goto fail;
    }

    vud_rank_nr_workers(&rank, 1);

    random_key(key);
    vud_ime_install_key(&rank, key, NULL, NULL);

    if (rank.err) {
        puts("cannot perform key exchange");
        goto fail;
    }

    puts("Key Exchange Done");

    vud_ime_launch(&rank);
    vud_ime_wait(&rank);

    printf("size,cycles,type,dpu\n");
    fetch_results_from(&rank);

    vud_rank_free(&rank);

    puts("Benchmark Finished.");
    return 0;

fail:
    if (rank.err) {
        printf("ime error: %s\n", vud_error_str(rank.err));
    }

    vud_rank_free(&rank);
    return 1;
}