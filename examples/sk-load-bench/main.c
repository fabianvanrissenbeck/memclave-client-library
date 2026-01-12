#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include <vud.h>
#include <vud_mem.h>
#include <vud_ime.h>

static uint64_t time_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);

    return ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static int perform_benchmark(bool auth_only) {
    vud_rank r = vud_rank_alloc(VUD_ALLOC_ANY);

    if (r.err) {
        goto error;
    }

    vud_ime_wait(&r);

    if (r.err) {
        goto error;
    }

    if (auth_only) {
        vud_ime_load_auth_only(&r, "../sk-load-bench");
    } else {
        vud_ime_load(&r, "../sk-load-bench");
    }

    if (r.err) {
        goto error;
    }

    uint64_t install_key_start = time_ms();
    vud_ime_install_key(&r, (uint8_t[32]) { 0 }, NULL, NULL);

    if (r.err) {
        goto error;
    }

    printf("Key Exchange took %lums.\n", time_ms() - install_key_start);

    vud_broadcast_to(&r, 8, &(uint64_t[8]) { 0 }, "__ime_debug_out");

    if (r.err) {
        goto error;
    }

    uint64_t launch_time = time_ms();
    vud_ime_launch(&r);

    if (r.err) {
        goto error;
    }

    vud_ime_wait(&r);

    if (r.err) {
        goto error;
    }

    printf("DPU finished in %lums.\n", time_ms() - launch_time);

    uint32_t buf[64][4];
    uint64_t* buf_ptr[64];

    for (int i = 0; i < 64; i++) {
        buf_ptr[i] = (uint64_t*) &buf[i][0];
    }

    vud_gather_from(&r, 2, "g_stats", &buf_ptr);

    if (r.err) {
        goto error;
    }

    printf("DPU,auth only,unload,auth,dec,scan\n");

    for (int i = 0; i < 64; i++) {
        printf("%02o,%s,%u,%u,%u,%u\n", i, auth_only ? "true" : "false", buf[i][0], buf[i][1], buf[i][2], buf[i][3]);
    }

    vud_rank_free(&r);
    return EXIT_SUCCESS;

error:
    printf("cannot run benchmark: %s\n", vud_error_str(r.err));
    return EXIT_FAILURE;
}

int main(void) {
    if (perform_benchmark(false) != 0) {
        return EXIT_FAILURE;
    }

    return perform_benchmark(true);
}