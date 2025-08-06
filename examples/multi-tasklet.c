#include "vud_ime.h"

#include <stdio.h>

int main(void) {
    vud_rank r = vud_rank_alloc(1);

    if (r.err) {
        puts("Cannot allocate rank.");
        return 1;
    }

    vud_ime_wait(&r);

    if (r.err) {
        puts("Cannot wait for rank.");
        return 1;
    }

    vud_broadcast_transfer(&r, 16, &(uint64_t[16]) { 0 }, 0x0);

    if (r.err) {
        puts("Cannot zero out prior results.");
        return 1;
    }

    vud_ime_launch_sk(&r, "../multi.sk");

    if (r.err) {
        printf("Cannot launch subkernel: vud error %d\n", r.err);
        return 1;
    }

    uint64_t data[64][16];
    uint64_t* data_ptr[64];

    for (int i = 0; i < 64; ++i) { data_ptr[i] = data[i]; }

    vud_ime_wait(&r);

    if (r.err) {
        puts("could not wait for subkernel completion");
        return 1;
    }

    vud_simple_gather(&r, 16, 0x0, &data_ptr);

    for (int j = 0; j < 64; ++j) {
        for (int i = 0; i < 16; ++i) {
            if (data[j][i] != i) {
                printf("Wrong result for DPU %d: Tasklet %d may not be running.\n", j, i);
            }
        }
    }

    vud_rank_free(&r);
    return 0;
}