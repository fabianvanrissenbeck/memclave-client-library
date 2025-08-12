#include <stdio.h>
#include <assert.h>
#include <vud_ime.h>

void buf_to_stdout(size_t sz, const uint64_t* buf) {
    FILE* p = popen("xxd -e -g 8", "w");
    assert(p != NULL);

    fwrite(buf, 1, sz * sizeof(buf[0]), p);
    pclose(p);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        puts("debug - Program to run a single subkernel and print the debug output.");
        puts("Usage: ./debug <subkernel>");

        return 1;
    }

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

    vud_mram_addr output_addr = (64 << 20) - 64;
    vud_broadcast_transfer(&r, 8, &(uint64_t[8]) { 0 }, output_addr);

    if (r.err) {
        puts("Cannot zero out prior results.");
        return 1;
    }

    vud_ime_launch_sk(&r, argv[1]);

    if (r.err) {
        printf("Cannot launch subkernel: vud error %d\n", r.err);
        return 1;
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
        printf("========== DPU %02o ==========\n", j);
        buf_to_stdout(8, &data[j][0]);
    }

    vud_rank_free(&r);
    return 0;
}