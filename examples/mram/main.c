/**
 * This benchmark measures the Guest->MRAM throughput with varying
 * block sizes and thread counts. Measurements are performed on one full rank.
 * No actual subkernels are loaded. We write into the lower 32-MiB of MRAM.
 * Outputs are formatted in CSV.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <inttypes.h>

#include <vud.h>
#include <vud_ime.h>

#define MIN_WORKER 1
#define MAX_WORKER 1

#define MIN_BLOCKS 64
#define MAX_BLOCKS (32 << 20)

#define N_ITER 1

static uint64_t s_clocks_per_sec;

static inline uint64_t rdtsc(void) {
    uint64_t res = 0;

    asm volatile(
        "mfence\n"
        "lfence\n"
        "rdtsc\n"
        "lfence\n"
        "shl $32, %%rdx\n"
        "or %%rdx, %%rax\n"
        "movq %%rax, %0\n"
        : "=r" (res)
    );

    return res;
}

static uint64_t* get_random_data(unsigned size) {
    FILE* fp = NULL;
    uint64_t* buffer = NULL;

    if ((fp = fopen("/dev/urandom", "rb")) == NULL) {
        goto cleanup;
    }

    if ((buffer = calloc(size / sizeof(uint64_t), sizeof(uint64_t))) == NULL) {
        goto cleanup;
    }

    if (fread(buffer, 1, size, fp) != size) {
        goto cleanup;
    }

    fclose(fp);
    return buffer;

cleanup:
    if (fp) { fclose(fp); }
    free(buffer);

    return NULL;
}

static void perform_benchmark_on(vud_rank* r, unsigned worker, unsigned size) {
    uint64_t* rand = get_random_data(size);
    if (rand == NULL) { return; }

    vud_rank_nr_workers(r, worker);

    if (r->err) { return; }

    uint64_t tm_start = rdtsc();
    vud_broadcast_transfer(r, size / sizeof(uint64_t), (const uint64_t (*)[]) rand, 0x0);
    uint64_t tm_end = rdtsc();

    uint64_t tm = tm_end - tm_start;
    double rate = ((double)(size * 64) / ((double) tm / (double)(s_clocks_per_sec))) / (double)(1024 * 1024);

    free(rand);
    printf("%u,%u,%" PRIu64",%"PRIu64",%.02f\n", size, worker + 1, tm, s_clocks_per_sec, rate);
}

int main(void) {
    vud_rank r = vud_rank_alloc(VUD_ALLOC_ANY);

    if (r.err) {
        printf("cannot allocate rank: %s\n", vud_error_str(r.err));
        return 1;
    }

    printf("size,threads,time (rdtsc),clocks per sec,transfer rate (MB/s)\n");

    uint64_t rdtsc_sec_start = rdtsc();
    sleep(1);
    uint64_t rdtsc_sec_end = rdtsc();

    s_clocks_per_sec = rdtsc_sec_end - rdtsc_sec_start;

    for (unsigned n_worker = MIN_WORKER; n_worker <= MAX_WORKER; ++n_worker) {
        for (unsigned block_size = MIN_BLOCKS; block_size <= MAX_BLOCKS; block_size = block_size * 2) {
            for (unsigned n = 0; n < N_ITER; ++n) {
                perform_benchmark_on(&r, n_worker, block_size);

                if (r.err) {
                    goto end;
                }
            }
        }
    }
end:

    if (r.err) {
        printf("could not perform benchmark: %s\n", vud_error_str(r.err));
        return 1;
    }

    vud_rank_free(&r);
}
