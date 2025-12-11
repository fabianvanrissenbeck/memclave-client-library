#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <inttypes.h>

#include <dpu.h>
#include <dpu_memory.h>
#include <dpu_transfer_matrix.h>

#define MIN_WORKER 0
#define MAX_WORKER 0 // upmem does not support >8 workers per rank

#define MIN_BLOCKS 64
#define MAX_BLOCKS (32 << 20)

#define N_ITER 5

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

static void perform_benchmark_on(struct dpu_rank_t* rank, unsigned worker, unsigned size) {
    uint64_t* rand = get_random_data(size);
    if (rand == NULL) { return; }

    struct dpu_transfer_matrix mat_bc = {
        .type = DPU_DEFAULT_XFER_MATRIX,
        .size = size,
        .offset = 0
    };

    for (int i = 0; i < 64; ++i) {
        mat_bc.ptr[i] = rand;
    }

    uint64_t tm_start = rdtsc();
    DPU_ASSERT(dpu_copy_to_mrams(rank, &mat_bc));
    uint64_t tm_end = rdtsc();

    uint64_t tm = tm_end - tm_start;
    double rate = ((double)(size * 64) / ((double) tm / (double)(s_clocks_per_sec))) / (double)(1024 * 1024);
    printf("broadcast,%u,%u,%" PRIu64",%"PRIu64",%.02f\n", size, worker, tm, s_clocks_per_sec, rate);

    uint64_t* buffer = calloc(64, size);

    if (buffer == NULL) {
        free(rand);
        return;
    }

    // touch all memory addresses to prevent measuring page allocation
    for (size_t i = 0; i < size * 64 / sizeof(uint64_t); ++i) {
        ((volatile uint64_t*) buffer)[i] = 0;
    }

    struct dpu_transfer_matrix mat_gt1 = {
        .type = DPU_DEFAULT_XFER_MATRIX,
        .size = size,
        .offset = 0,
    };

    tm_start = rdtsc();

    for (int i = 0; i < 64; ++i) {
        mat_gt1.ptr[i] = &buffer[i * size / sizeof(uint64_t)];
    }

    DPU_ASSERT(dpu_copy_from_mrams(rank, &mat_gt1));
    tm_end = rdtsc();

    tm = tm_end - tm_start;
    rate = ((double)(size * 64) / ((double) tm / (double)(s_clocks_per_sec))) / (double)(1024 * 1024);

    printf("gather,%u,%u,%" PRIu64",%"PRIu64",%.02f\n", size, worker, tm, s_clocks_per_sec, rate);

    for (int i = 0; i < 64; ++i) {
        if (memcmp(&buffer[i * size / sizeof(uint64_t)], rand, size) != 0) {
            printf("sanity check failure: broadcast/gather incorrect for DPU %d\n", i);

            free(rand);
            free(buffer);

            return;
        }
    }

    struct dpu_transfer_matrix mat_tf = {
        .type = DPU_DEFAULT_XFER_MATRIX,
        .size = size,
        .offset = 0,
    };

    tm_start = rdtsc();

    for (int i = 0; i < 64; ++i) {
        mat_tf.ptr[i] = &buffer[i * size / sizeof(uint64_t)];
    }

    DPU_ASSERT(dpu_copy_to_mrams(rank, &mat_tf));
    tm_end = rdtsc();

    tm = tm_end - tm_start;
    rate = ((double)(size * 64) / ((double) tm / (double)(s_clocks_per_sec))) / (double)(1024 * 1024);

    printf("transfer,%u,%u,%" PRIu64",%"PRIu64",%.02f\n", size, worker, tm, s_clocks_per_sec, rate);

    struct dpu_transfer_matrix mat_gt2 = {
        .type = DPU_DEFAULT_XFER_MATRIX,
        .size = size,
        .offset = 0,
    };

    tm_start = rdtsc();

    for (int i = 0; i < 64; ++i) {
        mat_gt2.ptr[i] = &buffer[i * size / sizeof(uint64_t)];
    }

    DPU_ASSERT(dpu_copy_from_mrams(rank, &mat_gt2));
    tm_end = rdtsc();


    tm = tm_end - tm_start;
    rate = ((double)(size * 64) / ((double) tm / (double)(s_clocks_per_sec))) / (double)(1024 * 1024);

    printf("gather,%u,%u,%" PRIu64",%"PRIu64",%.02f\n", size, worker, tm, s_clocks_per_sec, rate);

    for (int i = 0; i < 64; ++i) {
        if (memcmp(&buffer[i * size / sizeof(uint64_t)], rand, size) != 0) {
            printf("sanity check failure: transfer/gather incorrect for DPU %d\n", i);

            free(rand);
            free(buffer);

            return;
        }
    }

    free(rand);
    free(buffer);
}

uint64_t get_time_us(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);

    return ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

int main(void) {
    struct dpu_set_t set;

    printf("type,size,threads,time (rdtsc),clocks per sec,transfer rate (MB/s)\n");

    uint64_t time_us_start = get_time_us();
    uint64_t rdtsc_sec_start = rdtsc();
    sleep(1);
    uint64_t rdtsc_sec_end = rdtsc();
    uint64_t time_us_end = get_time_us();

    s_clocks_per_sec = (rdtsc_sec_end - rdtsc_sec_start) * 1000000 / (time_us_end - time_us_start);

    for (unsigned n_worker = MIN_WORKER; n_worker <= MAX_WORKER; ++n_worker) {
        char profile[120];

        if (n_worker) {
            snprintf(profile, sizeof(profile), "backend=hw,nrThreadsPerRank=%u", n_worker);
        } else {
            snprintf(profile, sizeof(profile), "backend=hw");
        }

        DPU_ASSERT(dpu_alloc_ranks(1, profile, &set));

        for (unsigned block_size = MIN_BLOCKS; block_size <= MAX_BLOCKS; block_size = block_size * 2) {
            for (unsigned n = 0; n < N_ITER; ++n) {
                perform_benchmark_on(set.list.ranks[0], n_worker, block_size);
            }
        }

        DPU_ASSERT(dpu_free(set));
    }
    return 0;
}