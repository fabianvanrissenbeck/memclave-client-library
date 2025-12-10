#include <stdio.h>
#include <assert.h>
#include <inttypes.h>

#include "vud.h"
#include "vud_ime.h"

#define MIN_BLOCK_SIZE 6
#define MAX_BLOCK_SIZE 24
#define SUBKERNEL "../chacha-bench"

static void random_key(uint8_t key[32]) {
    FILE* fp = fopen("/dev/urandom", "rb");

    assert(fp != NULL);
    assert(fread(key, 1, 32, fp) == 32);

    fclose(fp);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: crypto-bench <out scv>\n");
        return 1;
    }

    FILE* fp = fopen(argv[1], "w");

    if (fp == NULL) {
        perror("cannot create output file");
        return 1;
    }

    uint8_t key[32];
    vud_rank rank = vud_rank_alloc(VUD_ALLOC_ANY);

    vud_ime_wait(&rank);
    vud_ime_load(&rank, SUBKERNEL);
    random_key(key);
    vud_ime_install_key(&rank, key, NULL, NULL);
    vud_ime_launch(&rank);
    vud_ime_wait(&rank);

    if (rank.err) { goto fail; }

    uint64_t out_time_enc[64][MAX_BLOCK_SIZE - MIN_BLOCK_SIZE + 1];
    uint64_t out_time_dec[64][MAX_BLOCK_SIZE - MIN_BLOCK_SIZE + 1];

    uint64_t* ptr_time_enc[64];
    uint64_t* ptr_time_dec[64];

    for (int i = 0; i < 64; ++i) {
        ptr_time_enc[i] = &out_time_enc[i][0];
        ptr_time_dec[i] = &out_time_dec[i][0];
    }

    vud_gather_from(&rank, MAX_BLOCK_SIZE - MIN_BLOCK_SIZE + 1, "out_time_enc", &ptr_time_enc);
    vud_gather_from(&rank, MAX_BLOCK_SIZE - MIN_BLOCK_SIZE + 1, "out_time_dec", &ptr_time_dec);

    if (rank.err) { goto fail; }

    fprintf(fp, "size,cycles,type,dpu\n");

    for (int i = MIN_BLOCK_SIZE; i <= MAX_BLOCK_SIZE; ++i) {
        int idx = i - MIN_BLOCK_SIZE;

        for (int j = 0; j < 64; ++j) {
            fprintf(fp, "%zu,%" PRIu64 ",enc,%d\n", 1lu << i, out_time_enc[j][idx], j);
            fprintf(fp, "%zu,%" PRIu64 ",dec,%d\n", 1lu << i, out_time_dec[j][idx], j);
        }

    }

    fclose(fp);
    vud_rank_free(&rank);

    puts("Benchmark Finished.");
    return 0;

fail:
    if (rank.err) {
        printf("ime error: %s\n", vud_error_str(rank.err));
    }

    fclose(fp);
    vud_rank_free(&rank);

    return 1;
}