#include <vud.h>
#include <vud_sk.h>
#include <vud_ime.h>
#include <vud_mem.h>

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

    vud_rank_nr_workers(&r, 8);

    if (r.err) {
        printf("cannot increase worker count: %s\n", vud_error_str(r.err));
        goto error;
    }

    // only sets the location of the subkernel as of now
    // tihs is important to fetch symbol locations transparently
    // before launching

    vud_ime_load(&r, "../add");

    if (r.err) {
        puts("cannot load subkernel");
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
    uint64_t zero[64] = { 0 };
    uint64_t tgt_c[64];

    for (int i = 0; i < 64; ++i) {
        a[i] = i;
        b[i] = i;
        tgt_c[i] = a[i] + b[i];
    }

    vud_broadcast_to(&r, 64, &a, "a");
    vud_broadcast_to(&r, 64, &b, "b");
    vud_broadcast_to(&r, 64, &zero, "c");

    if (r.err) {
        puts("cannot transfer inputs");
        goto error;
    }

    vud_ime_launch(&r);

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

    vud_gather_from(&r, 64, "c", &c_ptr);

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