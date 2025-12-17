/**
 * This benchmark measures the Guest->MRAM throughput with varying
 * block sizes and thread counts. Measurements are performed on one full rank.
 * No actual subkernels are loaded. We write into the lower 32-MiB of MRAM.
 * Outputs are formatted in CSV. With a compile flag, one can run this benchmark
 * on UPMEM directly instead.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>

#if USE_UPMEM
#include <dpu.h>
#include <dpu_memory.h>
#include <dpu_transfer_matrix.h>
#else
#include <vud.h>
#include <vud_ime.h>
#endif

#if USE_UPMEM
// nrThreadsPerRank doesn't really have any effects
// but UPMEM does use 8 threads per rank as can be
// seen in the relevant source code
#define MIN_WORKER 8
#define MAX_WORKER 8
#else
#define MIN_WORKER 1
#define MAX_WORKER 16
#endif

#define MIN_BLOCKS 64
#define MAX_BLOCKS (32 << 20)

#define N_ITER 10

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

#if USE_UPMEM
static void perform_benchmark_on(struct dpu_rank_t* r, unsigned worker, unsigned size) {
#else
static void perform_benchmark_on(vud_rank* r, unsigned worker, unsigned size) {
#endif
    uint64_t* rand_1 = get_random_data(size);
    uint64_t* rand_2 = get_random_data(size);

    if (rand_1 == NULL || rand_2 == NULL) {
        free(rand_1);
        free(rand_2);

        return;
    }

#if !USE_UPMEM
    vud_rank_nr_workers(r, worker);

    if (r->err) {
        free(rand_1);
        free(rand_2);
        return;
    }
#endif

    uint64_t tm_start = UINT64_MAX;

    for (int i = -1; i < N_ITER; ++i) {
#if USE_UPMEM

        struct dpu_transfer_matrix mat = {
            .type = DPU_DEFAULT_XFER_MATRIX,
            .size = size,
            .offset = 0
        };

        for (int j = 0; j < 64; ++j) {
            mat.ptr[j] = rand_1;
        }

        DPU_ASSERT(dpu_copy_to_mrams(r, &mat));
#else
        vud_broadcast_transfer(r, size / sizeof(uint64_t), (const uint64_t (*)[]) rand_1, 0x0);
#endif

        // this forces one warm up cycle
        if (i == -1) {
            tm_start = rdtsc();
        }
    }

    uint64_t tm_end = rdtsc();

    uint64_t tm = (tm_end - tm_start) / N_ITER;
    double rate = ((double)(size * 64) / ((double) tm / (double)(s_clocks_per_sec))) / (double)(1024 * 1024);
    printf("broadcast,%u,%u,%" PRIu64",%"PRIu64",%.02f\n", size, worker, tm, s_clocks_per_sec, rate);

    uint64_t* buffer = calloc(64, size);

    if (buffer == NULL) {
        free(rand_1);
        free(rand_2);
        return;
    }

    // touch all memory addresses to prevent measuring page allocation
    for (size_t i = 0; i < size * 64 / sizeof(uint64_t); ++i) {
        ((volatile uint64_t*) buffer)[i] = 0;
    }

    for (int i = -1; i < N_ITER; ++i) {
#if USE_UPMEM
        struct dpu_transfer_matrix mat_gt1 = {
            .type = DPU_DEFAULT_XFER_MATRIX,
            .size = size,
            .offset = 0,
        };

        for (int j = 0; j < 64; ++j) {
            mat_gt1.ptr[j] = &buffer[j * size / sizeof(uint64_t)];
        }

        DPU_ASSERT(dpu_copy_from_mrams(r, &mat_gt1));
#else
        uint64_t* buffer_ptr[64];

        for (int j = 0; j < 64; ++j) {
            buffer_ptr[j] = &buffer[size * j / sizeof(uint64_t)];
        }

        vud_simple_gather(r, size / sizeof(uint64_t), 0x0, &buffer_ptr);

#endif
        if (i == -1) {
            tm_start = rdtsc();
        }
    }

    tm_end = rdtsc();

    tm = (tm_end - tm_start) / N_ITER;
    rate = ((double)(size * 64) / ((double) tm / (double)(s_clocks_per_sec))) / (double)(1024 * 1024);

    printf("gather,%u,%u,%" PRIu64",%"PRIu64",%.02f\n", size, worker, tm, s_clocks_per_sec, rate);

    for (int i = 0; i < 64; ++i) {
        if (memcmp(&buffer[i * size / sizeof(uint64_t)], rand_1, size) != 0) {
            printf("sanity check failure: broadcast/gather incorrect for DPU %d\n", i);

            free(rand_1);
            free(rand_2);
            free(buffer);

            return;
        }

        memcpy(&buffer[i * size / sizeof(uint64_t)], rand_2, size);
    }

    for (int i = -1; i < N_ITER; ++i) {
#if USE_UPMEM
        struct dpu_transfer_matrix mat_gt1 = {
            .type = DPU_DEFAULT_XFER_MATRIX,
            .size = size,
            .offset = 0,
        };

        for (int j = 0; j < 64; ++j) {
            mat_gt1.ptr[j] = &buffer[j * size / sizeof(uint64_t)];
        }

        DPU_ASSERT(dpu_copy_to_mrams(r, &mat_gt1));
#else
        uint64_t* buffer_ptr[64];

        for (int j = 0; j < 64; ++j) {
            buffer_ptr[j] = &buffer[size * j / sizeof(uint64_t)];
        }

        vud_simple_transfer(r, size / sizeof(uint64_t), &buffer_ptr, 0x0);

#endif
        if (i == -1) {
            tm_start = rdtsc();
        }
    }

    tm_end = rdtsc();

    tm = (tm_end - tm_start) / N_ITER;
    rate = ((double)(size * 64) / ((double) tm / (double)(s_clocks_per_sec))) / (double)(1024 * 1024);

    printf("transfer,%u,%u,%" PRIu64",%"PRIu64",%.02f\n", size, worker, tm, s_clocks_per_sec, rate);

    for (int i = -1; i < N_ITER; ++i) {
#if USE_UPMEM
        struct dpu_transfer_matrix mat_gt1 = {
            .type = DPU_DEFAULT_XFER_MATRIX,
            .size = size,
            .offset = 0,
        };

        for (int j = 0; j < 64; ++j) {
            mat_gt1.ptr[j] = &buffer[j * size / sizeof(uint64_t)];
        }

        DPU_ASSERT(dpu_copy_from_mrams(r, &mat_gt1));
#else
        uint64_t* buffer_ptr[64];

        for (int j = 0; j < 64; ++j) {
            buffer_ptr[j] = &buffer[size * j / sizeof(uint64_t)];
        }

        vud_simple_gather(r, size / sizeof(uint64_t), 0x0, &buffer_ptr);

#endif
        if (i == -1) {
            tm_start = rdtsc();
        }
    }

    tm_end = rdtsc();

    tm = (tm_end - tm_start) / N_ITER;
    rate = ((double)(size * 64) / ((double) tm / (double)(s_clocks_per_sec))) / (double)(1024 * 1024);

    printf("gather,%u,%u,%" PRIu64",%"PRIu64",%.02f\n", size, worker, tm, s_clocks_per_sec, rate);

    for (int i = 0; i < 64; ++i) {
        if (memcmp(&buffer[i * size / sizeof(uint64_t)], rand_2, size) != 0) {
            printf("sanity check failure: transfer/gather incorrect for DPU %d\n", i);

            free(rand_1);
            free(rand_2);
            free(buffer);

            return;
        }
    }

    free(rand_1);
    free(rand_2);
    free(buffer);
}

int main(void) {
#if USE_UPMEM
    struct dpu_set_t set;
    struct dpu_rank_t* r;

    DPU_ASSERT(dpu_alloc_ranks(1, "backend=hw", &set));
    r = set.list.ranks[0];
#else
    vud_rank r = vud_rank_alloc(VUD_ALLOC_ANY);
    vud_ime_wait(&r); // necessary - otherwise the CI-switch state may be incorrect

    if (r.err) {
        printf("cannot allocate rank: %s\n", vud_error_str(r.err));
        return 1;
    }
#endif

    printf("type,size,threads,time (rdtsc),clocks per sec,transfer rate (MB/s)\n");

    uint64_t rdtsc_sec_start = rdtsc();
    sleep(1);
    uint64_t rdtsc_sec_end = rdtsc();

    s_clocks_per_sec = rdtsc_sec_end - rdtsc_sec_start;

    for (unsigned n_worker = MIN_WORKER; n_worker <= MAX_WORKER; ++n_worker) {
        for (unsigned block_size = MIN_BLOCKS; block_size <= MAX_BLOCKS; block_size = block_size * 2) {
#if USE_UPMEM
            perform_benchmark_on(r, n_worker, block_size);
#else
            perform_benchmark_on(&r, n_worker, block_size);

            if (r.err) {
                goto end;
            }
#endif
        }
    }

#if USE_UPMEM
    DPU_ASSERT(dpu_free(set));
#else
end:

    if (r.err) {
        printf("could not perform benchmark: %s\n", vud_error_str(r.err));
        return 1;
    }

    vud_rank_free(&r);
#endif
}
