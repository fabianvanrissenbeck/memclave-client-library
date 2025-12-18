#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>    // for getopt() & optind
#include <getopt.h>    // for getopt_long(), if you ever need it
#include <stdbool.h>

#define NR_DPUS 64

#define DPU_BINARY "../gemv"

#include <vud.h>
#include <vud_sk.h>
#include <vud_ime.h>
#include <vud_mem.h>
#include "../../src/vud_log.h"
#include "support/common.h"    // defines T, dpu_arguments_t, dpu_results_t, etc.
#include "support/params.h"    // parses command-line into struct Params
#include "support/timer.h"
#include "support/prim_results.h"

/* MRAM layout */
#ifndef ARG_OFFSET
#define ARG_OFFSET 0x2000u
#endif
#ifndef ARG_SIZE
#define ARG_SIZE   (sizeof(dpu_arguments_t))
#endif
#ifndef A_OFFSET
#define A_OFFSET   (ARG_OFFSET + ((ARG_SIZE + 0xFFu) & ~0xFFu))
#endif

#define MRAM_SIZE_BYTES     (64u << 20)
#define SK_LOG_SIZE_BYTES   64
#define SK_LOG_OFFSET       (MRAM_SIZE_BYTES - SK_LOG_SIZE_BYTES)

#define LOG_WORDS 8
#define LOG_MAGIC 0x534B4C4F475631ULL /* "SKLOGV1" */

static T* A;
static T* B;
static T* C;
static T* C_dpu;

static void random_key(uint8_t key[32]) {
    FILE* fp = fopen("/dev/urandom", "rb");

    assert(fp != NULL);
    assert(fread(key, 1, 32, fp) == 32);

    fclose(fp);
}

/* Create input arrays */
static void init_data(T* A_, T* B_, unsigned int m_size, unsigned int n_size) {
    srand(0);
    for (unsigned int i = 0; i < m_size * n_size; i++) {
        A_[i] = (unsigned int)(rand() % 50);
    }
    for (unsigned int i = 0; i < n_size; i++) {
        B_[i] = (unsigned int)(rand() % 50);
    }
}

/* Compute output on the host */
static void gemv_host(T* C_, T* A_, T* B_, unsigned int m_size, unsigned int n_size) {
    for (unsigned int i = 0; i < m_size; i++) C_[i] = 0;
    for (unsigned int m = 0; m < m_size; m++) {
        for (unsigned int n = 0; n < n_size; n++) {
            C_[m] += A_[m * n_size + n] * B_[n];
        }
    }
}

static void push_args_array(vud_rank* r, dpu_arguments_t* args, uint32_t nr_of_dpus) {
    const size_t words = (sizeof(dpu_arguments_t) + 7u) / 8u;

    /* Up to 64B args (8 qwords) – plenty for typical GEMV args */
    enum { WORDS_MAX = 8 };
    assert(words <= WORDS_MAX);
    assert(nr_of_dpus <= NR_DPUS);

    _Alignas(8) uint64_t staged[NR_DPUS][WORDS_MAX];
    for (uint32_t i = 0; i < nr_of_dpus; ++i) {
        memset(staged[i], 0, words * 8u);
        memcpy(staged[i], &args[i], sizeof(dpu_arguments_t));
    }

    const uint64_t* lanes[NR_DPUS];
    for (uint32_t i = 0; i < nr_of_dpus; ++i) lanes[i] = staged[i];
    for (uint32_t i = nr_of_dpus; i < NR_DPUS; ++i) lanes[i] = staged[0];

    vud_simple_transfer(r, (vud_mram_size)words, (const uint64_t* (*)[NR_DPUS])&lanes, (vud_mram_addr)ARG_OFFSET);
}

int main(int argc, char** argv) {
    struct Params p = input_params(argc, argv);

    vud_rank r = vud_rank_alloc(VUD_ALLOC_ANY);
    if (r.err) { 
	    fprintf(stderr, "vud_rank_alloc failed: %s\n", vud_error_str(r.err)); 
	    return EXIT_FAILURE; 
    }

    vud_ime_wait(&r);
    if (r.err) { 
	    fprintf(stderr, "vud_ime_wait failed: %s\n", vud_error_str(r.err)); 
	    return EXIT_FAILURE; 
    }

    vud_ime_load(&r, DPU_BINARY);
    if (r.err) { 
	    fprintf(stderr, "cannot load subkernel: %s\n", vud_error_str(r.err)); 
	    return EXIT_FAILURE; 
    }

    vud_rank_nr_workers(&r, 12);
    if (r.err) { 
	    fprintf(stderr, "cannot start worker threads: %s\n", vud_error_str(r.err)); 
	    return EXIT_FAILURE; 
    }

    uint8_t key[32];
    random_key(key);

    //vud_ime_install_key(&r, key, NULL, NULL);

    if (r.err) {
        puts("key exchange failed");
        return EXIT_FAILURE;
    }

#if ENERGY
	struct dpu_probe_t probe;
	DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif
    const uint32_t nr_of_dpus = NR_DPUS;

    unsigned int m_size = p.m_size;
    unsigned int n_size = p.n_size;

    /* helpers */
    dpu_info = (struct dpu_info_t*)malloc(nr_of_dpus * sizeof(struct dpu_info_t));
    dpu_arguments_t* input_args = (dpu_arguments_t*)malloc(nr_of_dpus * sizeof(dpu_arguments_t));
    if (!dpu_info || !input_args) { fprintf(stderr, "malloc failed\n"); return EXIT_FAILURE; }

    uint32_t max_rows_per_dpu = 0;

    uint32_t n_size_pad = n_size;
    if (n_size % 2 == 1) n_size_pad++;

    /* partitioning over rows */
    for (uint32_t i = 0; i < nr_of_dpus; i++) {
        uint32_t rows_per_dpu;
        uint32_t prev_rows_dpu = 0;

        uint32_t chunks = m_size / nr_of_dpus;
        rows_per_dpu = chunks;

        uint32_t rest_rows = m_size % nr_of_dpus;
        if (i < rest_rows) rows_per_dpu++;

        if (rest_rows > 0) {
            if (i >= rest_rows)
                prev_rows_dpu = rest_rows * (chunks + 1) + (i - rest_rows) * chunks;
            else
                prev_rows_dpu = i * (chunks + 1);
        } else {
            prev_rows_dpu = i * chunks;
        }

        /* padding of rows (for 4-byte elements) */
        uint32_t rows_per_dpu_pad = rows_per_dpu;
        if (rows_per_dpu_pad % 2 == 1) rows_per_dpu_pad++;
        if (rows_per_dpu_pad > max_rows_per_dpu) max_rows_per_dpu = rows_per_dpu_pad;

        dpu_info[i].rows_per_dpu      = rows_per_dpu;
        dpu_info[i].rows_per_dpu_pad  = rows_per_dpu_pad;
        dpu_info[i].prev_rows_dpu     = prev_rows_dpu;

        /* per-DPU args (max_rows filled later each rep) */
        input_args[i].n_size     = n_size;
        input_args[i].n_size_pad = n_size_pad;
        input_args[i].nr_rows    = rows_per_dpu;
        input_args[i].max_rows   = 0;
    }

    /* Host buffers */
    A = (T*)malloc((size_t)max_rows_per_dpu * nr_of_dpus * n_size_pad * sizeof(T));
    B = (T*)malloc((size_t)n_size_pad * sizeof(T));
    C = (T*)malloc((size_t)max_rows_per_dpu * nr_of_dpus * sizeof(T));
    C_dpu = (T*)malloc((size_t)max_rows_per_dpu * nr_of_dpus * sizeof(T));
    if (!A || !B || !C || !C_dpu) { fprintf(stderr, "malloc failed\n"); return EXIT_FAILURE; }

    init_data(A, B, m_size, n_size);

    /* MRAM layout per DPU */
    const uint32_t slice_bytes = max_rows_per_dpu * n_size_pad * sizeof(T);
    const uint32_t vec_bytes   = n_size_pad * sizeof(T);

    const vud_mram_addr mram_A = (vud_mram_addr)A_OFFSET;
    const vud_mram_addr mram_B = (vud_mram_addr)(A_OFFSET + slice_bytes);
    const vud_mram_addr mram_C = (vud_mram_addr)(A_OFFSET + slice_bytes + vec_bytes);

    /* safety vs end-of-MRAM log */
    const uint64_t result_bytes = (uint64_t)max_rows_per_dpu * (uint64_t)sizeof(T);
    assert((uint64_t)mram_C + result_bytes <= (uint64_t)SK_LOG_OFFSET);

    /* Timer */
    Timer timer;

    /* CPU reference once */
    start(&timer, 0, 0);
    gemv_host(C, A, B, m_size, n_size);
    stop(&timer, 0);

    for (unsigned int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        /* CPU->DPU transfers */
        if (rep >= p.n_warmup) start(&timer, 1, rep - p.n_warmup);

        for (uint32_t i = 0; i < nr_of_dpus; i++) {
            input_args[i].max_rows = max_rows_per_dpu; /* set before push */
        }

        /* 1) push args */
        push_args_array(&r, input_args, nr_of_dpus);
        if (r.err) { fprintf(stderr, "push args failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

        /* 2) scatter matrix slice A */
        {
            assert((slice_bytes % 8u) == 0);
            const vud_mram_size slice_words = (vud_mram_size)(slice_bytes / 8u);

            const uint64_t* lanes[NR_DPUS];
            for (uint32_t i = 0; i < nr_of_dpus; ++i) {
                lanes[i] = (const uint64_t*)(A + (size_t)dpu_info[i].prev_rows_dpu * n_size);
            }
            for (uint32_t i = nr_of_dpus; i < NR_DPUS; ++i) lanes[i] = lanes[0];

            vud_simple_transfer(&r, slice_words, (const uint64_t* (*)[NR_DPUS])&lanes, mram_A);
            if (r.err) { fprintf(stderr, "push A failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
        }

        /* 3) broadcast vector B */
        {
            assert((vec_bytes % 8u) == 0);
            const vud_mram_size vec_words = (vud_mram_size)(vec_bytes / 8u);

            const uint64_t* lanes[NR_DPUS];
            for (uint32_t i = 0; i < nr_of_dpus; ++i) lanes[i] = (const uint64_t*)B;
            for (uint32_t i = nr_of_dpus; i < NR_DPUS; ++i) lanes[i] = lanes[0];

            vud_simple_transfer(&r, vec_words, (const uint64_t* (*)[NR_DPUS])&lanes, mram_B);
            if (r.err) { fprintf(stderr, "push B failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
        }

        if (rep >= p.n_warmup) stop(&timer, 1);

        /* launch */
        if (rep >= p.n_warmup) {
            start(&timer, 2, rep - p.n_warmup);
#if ENERGY
            DPU_ASSERT(dpu_probe_start(&probe));
#endif
        }

        vud_ime_launch(&r);
        if (r.err) { fprintf(stderr, "launch failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
        vud_ime_wait(&r);
        if (r.err) { fprintf(stderr, "wait failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

        if (rep >= p.n_warmup) {
            stop(&timer, 2);
#if ENERGY
            DPU_ASSERT(dpu_probe_stop(&probe));
#endif
        }

        /* allow host MRAM access */
        vud_rank_rel_mux(&r);
        vud_ime_wait(&r);
        if (r.err) { fprintf(stderr, "post-rel-mux wait failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

        /* print DPU cycle max (optional, but useful like VA) */
        {
            uint64_t logbuf[NR_DPUS][LOG_WORDS];
            uint64_t* lanes[NR_DPUS];
            for (uint32_t d = 0; d < nr_of_dpus; ++d) lanes[d] = logbuf[d];
            for (uint32_t d = nr_of_dpus; d < NR_DPUS; ++d) lanes[d] = logbuf[0];

            vud_simple_gather(&r, (vud_mram_size)LOG_WORDS, (vud_mram_addr)SK_LOG_OFFSET,
                              (uint64_t* (*)[NR_DPUS])&lanes);
            if (r.err) { fprintf(stderr, "gather log failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

            uint64_t max_cycles = 0;
            for (uint32_t d = 0; d < nr_of_dpus; ++d) {
                if (logbuf[d][0] == LOG_MAGIC && logbuf[d][7] == 1ULL) {
                    if (logbuf[d][1] > max_cycles) max_cycles = logbuf[d][1];
                }
            }
            printf("DPU cycles (whole-kernel, max over DPUs): %llu\n",
                   (unsigned long long)max_cycles);
        }

        /* retrieve results */
        if (rep >= p.n_warmup) start(&timer, 3, rep - p.n_warmup);
        {
            const uint32_t out_bytes = max_rows_per_dpu * sizeof(T);
            assert((out_bytes % 8u) == 0);
            const vud_mram_size out_words = (vud_mram_size)(out_bytes / 8u);

            uint64_t* lanes[NR_DPUS];
            for (uint32_t i = 0; i < nr_of_dpus; ++i) {
                lanes[i] = (uint64_t*)(C_dpu + (size_t)i * max_rows_per_dpu);
            }
            for (uint32_t i = nr_of_dpus; i < NR_DPUS; ++i) lanes[i] = lanes[0];

            vud_simple_gather(&r, out_words, mram_C, (uint64_t* (*)[NR_DPUS])&lanes);
            if (r.err) { fprintf(stderr, "gather C failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
        }
        if (rep >= p.n_warmup) stop(&timer, 3);

    }
#if ENERGY
	double acc_energy, avg_energy, acc_time, avg_time;
	DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_ACCUMULATE, &acc_energy));
	DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &avg_energy));
	DPU_ASSERT(dpu_probe_get(&probe, DPU_TIME, DPU_ACCUMULATE, &acc_time));
	DPU_ASSERT(dpu_probe_get(&probe, DPU_TIME, DPU_AVERAGE, &avg_time));
#endif

	// Print timing results
	printf("CPU Version Time (ms): ");
	print(&timer, 0, 1);
	printf("CPU-DPU Time (ms): ");
	print(&timer, 1, p.n_reps);
	printf("DPU Kernel Time (ms): ");
	print(&timer, 2, p.n_reps);
	printf("DPU-CPU Time (ms): ");
	print(&timer, 3, p.n_reps);
	printf("\n");
        // update CSV
#define TEST_NAME "GEMV"
#define RESULTS_FILE "prim_results.csv"
        //update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 0, p.n_reps, "CPU");
        update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 1, p.n_reps, "M_C2D");
        update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 3, p.n_reps, "M_D2C");
        update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 2, p.n_reps, "DPU");

#if ENERGY
	printf("Energy (J): %f J\t", avg_energy);
#endif

	// Check output
    bool status = true;
    unsigned int n, j;
    unsigned int idx = 0;

    for (n = 0; n < nr_of_dpus; n++) {
        for (j = 0; j < dpu_info[n].rows_per_dpu; j++) {
            if (C[idx] != C_dpu[n * max_rows_per_dpu + j]) {
                status = false;
                printf("%u: %u -- %u\n", idx, (unsigned)C[idx], (unsigned)C_dpu[n * max_rows_per_dpu + j]);
                break;
            }
            idx++;
        }
        if (!status) break;
    }

    if (status) printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    else        printf("[" ANSI_COLOR_RED   "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");

	// Deallocation
	free(A);
	free(B);
	free(C);
	free(C_dpu);
        vud_rank_free(&r);
        return 0;

#if ENERGY
	DPU_ASSERT(dpu_probe_deinit(&probe));
#endif
}
