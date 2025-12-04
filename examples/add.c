#include <vud.h>
#include <vud_mem.h>
#include <vud_ime.h>

#include <stdio.h>
#include <assert.h>

static void random_key(uint8_t key[32]) {
    FILE* fp = fopen("/dev/urandom", "rb");

    assert(fp != NULL);
    assert(fread(key, 1, 32, fp) == 32);

    fclose(fp);
}

int main(void) {
    vud_rank r = vud_rank_alloc(VUD_ALLOC_ANY);

    if (r.err) {
        puts("Cannot allocate rank.");
        return 1;
    }

    vud_ime_wait(&r);

    if (r.err) {
        puts("cannot wait for rank");
        goto error;
    }

    uint8_t key[32];
    random_key(key);

    vud_ime_install_key(&r, key, NULL, NULL);

    if (r.err) {
        puts("key exchange failed");
        goto error;
    }

    uint64_t a[64];
    uint64_t b[64];
    uint64_t tgt_c[64];

    for (int i = 0; i < 64; ++i) {
        a[i] = i;
        b[i] = 2 * i;
        tgt_c[i] = a[i] + b[i];
    }

    vud_broadcast_transfer(&r, 64, &a, 0x0);
    vud_broadcast_transfer(&r, 64, &b, sizeof(a));

    if (r.err) {
        puts("cannot transfer inputs");
        goto error;
    }

    vud_ime_launch(&r, "../add");

    if (r.err) {
        puts("failed to launch subkernel");
        goto error;
    }

    vud_ime_wait(&r);

    if (r.err) {
        puts("could not wait for subkernel completion");
        goto error;
    }

    uint64_t c[64][64];
    uint64_t* c_ptr[64];

    for (int i = 0; i < 64; ++i) { c_ptr[i] = &c[i][0]; }

    vud_simple_gather(&r, 64, sizeof(a) + sizeof(b), &c_ptr);

    int errors = 0;

    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j) {
            if (c[i][j] != tgt_c[j]) {
                printf("[DPU %02o] Value of c[%02d] is %016lx. Should be %02lx.\n", i, j, c[i][j], tgt_c[j]);
                errors += 1;
            }
        }
    }

    if (errors) {
        printf("Test finished with %d errors.\n", errors);
    } else {
        printf("Test finished successfully.\n");
    }

    vud_rank_free(&r);
    return 0;

error:
    printf("VUD Error %s\n", vud_error_str(r.err));
    vud_rank_free(&r);
    return 1;
}