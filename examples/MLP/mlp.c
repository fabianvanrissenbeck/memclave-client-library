/**
 * app.c
 * MLP Host Application Source File
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <time.h>
#include <inttypes.h>

#if ENERGY
#include <dpu_probe.h>
#endif

#define NR_DPUS 64

#include <vud.h>
#include <vud_sk.h>
#include <vud_ime.h>
#include <vud_mem.h>
#include "../../src/vud_log.h"
#include "support/common.h"    // defines T, dpu_arguments_t, dpu_results_t, etc.
#include "support/params.h"    // parses command-line into struct Params
#include "support/timer.h"     // for timing host vs DPU if you want
#include "support/prim_results.h" 

/* MRAM layout */
#ifndef ARG_OFFSET
#define ARG_OFFSET 0x2000u
#endif
#ifndef ARG_SIZE
#define ARG_SIZE   (sizeof(dpu_arguments_t))
#endif
#ifndef A_OFFSET
//#define ALIGN_UP(x,a) (((x)+(a)-1) & ~((a)-1))
//#define A_OFFSET ALIGN_UP(ARG_OFFSET + ARG_SIZE, 0x2000u)
#define A_OFFSET   (ARG_OFFSET + ((ARG_SIZE + 0xFFu) & ~0xFFu))
#endif

#define MRAM_SIZE_BYTES     (64u << 20)
#define SK_LOG_SIZE_BYTES   64u
#define SK_LOG_OFFSET       (MRAM_SIZE_BYTES - SK_LOG_SIZE_BYTES)

#ifndef DPU_BINARY
#define DPU_BINARY "../mlp"
#endif

#define CTRL_OFFSET (ARG_OFFSET + 0x40u)

typedef struct __attribute__((aligned(8))) {
    uint32_t cmd;      // 0=IDLE, 1=RUN, 2=EXIT
    uint32_t job_id;
    uint32_t status;   // 0=WAITING, 1=RUNNING, 2=DONE, 3=EXITED
    uint32_t _pad;
} ctrl_t;

#define CMD_IDLE 0
#define CMD_RUN  1
#define CMD_EXIT 2
#define ST_WAITING 0
#define ST_RUNNING 1
#define ST_DONE    2
#define ST_EXITED  3

static T** A;
static T* B;
static T* B_host;
static T* B_tmp;
static T* C;
static T* C_dpu;

// Create input arrays
static void init_data(T** A, T* B, T* B_host, unsigned int m_size, unsigned int n_size) {
	for (unsigned int l = 0; l < NUM_LAYERS; l++)
		for (unsigned int i = 0; i < m_size * n_size; i++){
			if(i % 100 < 98){
				A[l][i] = 0;
			}else{
				A[l][i] = (l+i) % 2;
			}
		}
	for (unsigned int i = 0; i < n_size; i++){
		if(i % 50 < 48){
			B[i] = 0;
		}
		else{
			B[i] = i % 2;
		}
		B_host[i] = B[i];
	}
}

// Compute output in the host
static void mlp_host(T* C, T** A, T* B, unsigned int m_size, unsigned int n_size) {

	for (unsigned int nl = 0; nl < NUM_LAYERS; nl++){
		for (unsigned int m = 0; m < m_size; m++){
			C[m] = 0;
		}
		for (unsigned int m = 0; m < m_size; m++){
			for (unsigned int n = 0; n < n_size; n++){
				C[m] += A[nl][m * n_size + n] * B[n];
			}
			C[m] = max(0, C[m]);
		}
		for (unsigned int n = 0; n < n_size; n++){
			B[n] = C[n];
		}
	}
}

static void push_args_array(vud_rank* r, const dpu_arguments_t* args, uint32_t nr_of_dpus) {
    const size_t words = (sizeof(dpu_arguments_t) + 7u) / 8u;
    const size_t lane_bytes = words * 8u;

    void* staged = NULL;
    if (posix_memalign(&staged, 8, NR_DPUS * lane_bytes) != 0) {
        fprintf(stderr, "push_args_array: staging alloc failed\n");
        exit(1);
    }
    memset(staged, 0, NR_DPUS * lane_bytes);

    for (uint32_t i = 0; i < nr_of_dpus; i++) {
        memcpy((uint8_t*)staged + (size_t)i * lane_bytes, &args[i], sizeof(dpu_arguments_t));
    }

    const uint64_t* lanes[NR_DPUS];
    for (uint32_t i = 0; i < nr_of_dpus; i++) lanes[i] = (const uint64_t*)((uint8_t*)staged + (size_t)i * lane_bytes);
    for (uint32_t i = nr_of_dpus; i < NR_DPUS; i++) lanes[i] = lanes[0];

    vud_simple_transfer(r, (vud_mram_size)words, (const uint64_t* (*)[NR_DPUS])&lanes, (vud_mram_addr)ARG_OFFSET);
    free(staged);
}

static void push_ctrl_broadcast(vud_rank *r, ctrl_t ctrl) {
    const vud_mram_size words = (vud_mram_size)((sizeof(ctrl_t) + 7u) / 8u);
    _Alignas(8) uint64_t staged[NR_DPUS][2];
    const uint64_t *lanes[NR_DPUS];

    for (uint32_t i = 0; i < NR_DPUS; i++) {
        memset(staged[i], 0, words * 8u);
        memcpy(staged[i], &ctrl, sizeof(ctrl_t));
        lanes[i] = staged[i];
    }
    vud_simple_transfer(r, words, (const uint64_t (*)[NR_DPUS])&lanes, (vud_mram_addr)CTRL_OFFSET);
}


// Main of the Host Application
int main(int argc, char **argv) {

    struct Params p = input_params(argc, argv);

    const uint32_t nr_of_dpus = NR_DPUS;
    unsigned int m_size = p.m_size;
    unsigned int n_size = p.n_size;

    vud_rank r = vud_rank_alloc(VUD_ALLOC_ANY);
    if (r.err) { fprintf(stderr, "rank_alloc failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

    vud_ime_wait(&r);
    if (r.err) { fprintf(stderr, "ime_wait failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

    vud_ime_load(&r, DPU_BINARY);
    if (r.err) { fprintf(stderr, "cannot load subkernel: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

    vud_rank_nr_workers(&r, 12);
    if (r.err) { fprintf(stderr, "cannot start worker threads: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

    uint8_t key[32];
    //random_key(key);
 
    //vud_ime_install_key(&r, key, NULL, NULL);
 
    if (r.err) {
        puts("key exchange failed");
        return -1;
    }

    // Initialize help data
    dpu_info = (struct dpu_info_t *)malloc(nr_of_dpus * sizeof(struct dpu_info_t));
    dpu_arguments_t *input_args = (dpu_arguments_t *)malloc(nr_of_dpus * sizeof(dpu_arguments_t));
    if (!dpu_info || !input_args) { fprintf(stderr, "malloc failed\n"); return EXIT_FAILURE; }

    uint32_t max_rows_per_dpu = 0;
    uint32_t n_size_pad = n_size;
    if (n_size % 2 == 1) n_size_pad++; 

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

        uint32_t rows_per_dpu_pad = rows_per_dpu;
        if (rows_per_dpu_pad % 2 == 1) rows_per_dpu_pad++;
        if (rows_per_dpu_pad > max_rows_per_dpu) max_rows_per_dpu = rows_per_dpu_pad;

        dpu_info[i].rows_per_dpu     = rows_per_dpu;
        dpu_info[i].rows_per_dpu_pad = rows_per_dpu_pad;
        dpu_info[i].prev_rows_dpu    = prev_rows_dpu;

        input_args[i].n_size     = n_size;
        input_args[i].n_size_pad = n_size_pad;
        input_args[i].nr_rows    = rows_per_dpu;
        input_args[i].max_rows   = 0; /* set per rep like PRiM */
    }

    /* buffers (safe: allocate B/B_host with n_size_pad) */
    A = (T**)malloc(NUM_LAYERS * sizeof(T*));
    for (unsigned int l = 0; l < NUM_LAYERS; l++) {
        A[l] = (T*)malloc((size_t)max_rows_per_dpu * nr_of_dpus * n_size_pad * sizeof(T));
        if (!A[l]) { fprintf(stderr, "malloc A[%u] failed\n", l); return EXIT_FAILURE; }
    }

    B      = (T*)malloc((size_t)n_size_pad * sizeof(T));
    B_host = (T*)malloc((size_t)n_size_pad * sizeof(T));
    C      = (T*)malloc((size_t)m_size * sizeof(T));
    C_dpu  = (T*)malloc((size_t)max_rows_per_dpu * nr_of_dpus * sizeof(T));
    B_tmp  = (T*)malloc((size_t)max_rows_per_dpu * nr_of_dpus * sizeof(T));

    if (!B || !B_host || !C || !C_dpu || !B_tmp) { fprintf(stderr, "malloc failed\n"); return EXIT_FAILURE; }
    memset(B, 0, (size_t)n_size_pad * sizeof(T));
    memset(B_host, 0, (size_t)n_size_pad * sizeof(T));

    init_data(A, B, B_host, m_size, n_size);
    if (n_size_pad > n_size) { B[n_size] = 0; B_host[n_size] = 0; }

    /* MRAM offsets (PRiM layout) */
    const size_t slice_bytes = (size_t)max_rows_per_dpu * n_size_pad * sizeof(T);
    const size_t vec_bytes   = (size_t)n_size_pad * sizeof(T);

    const vud_mram_addr A_off = (vud_mram_addr)A_OFFSET;
    const vud_mram_addr B_off = (vud_mram_addr)(A_OFFSET + slice_bytes);
    const vud_mram_addr C_off = (vud_mram_addr)(A_OFFSET + slice_bytes + vec_bytes);

    assert((slice_bytes % 8u) == 0);
    assert((vec_bytes % 8u) == 0);

    const vud_mram_size slice_words = (vud_mram_size)(slice_bytes / 8u);
    const vud_mram_size vec_words   = (vud_mram_size)(vec_bytes / 8u);
    const vud_mram_size c_words     = (vud_mram_size)((max_rows_per_dpu * sizeof(T)) / 8u);


    /* CPU reference */
    Timer timer;
    start(&timer, 0, 0);
    mlp_host(C, A, B_host, m_size, n_size);
    stop(&timer, 0);


    ctrl_t c = { .cmd=CMD_RUN, .job_id=1, .status=ST_WAITING, ._pad=0 };
    for (unsigned int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        c.cmd=CMD_RUN; c.job_id=1; c.status=ST_WAITING; c._pad=0;
        push_ctrl_broadcast(&r, c);
        vud_ime_launch(&r);
        if (r.err) { fprintf(stderr, "vud_ime_launch failed %d\n", r.err); return -1; }
        vud_ime_wait(&r);
        if (r.err) { fprintf(stderr, "vud_ime_wait failed\n"); return -1; }
 

        if (rep >= p.n_warmup) start(&timer, 1, rep - p.n_warmup);

        /* args + first layer data */
        for (uint32_t i = 0; i < nr_of_dpus; i++) input_args[i].max_rows = max_rows_per_dpu;
        push_args_array(&r, input_args, nr_of_dpus);
        if (r.err) { fprintf(stderr, "push args failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

        /* scatter A[0] slices */
        {
            const uint64_t* lanes[NR_DPUS];
            for (uint32_t i = 0; i < nr_of_dpus; i++) {
                lanes[i] = (const uint64_t*)(A[0] + (size_t)dpu_info[i].prev_rows_dpu * n_size);
            }
            for (uint32_t i = nr_of_dpus; i < NR_DPUS; i++) lanes[i] = lanes[0];

            vud_simple_transfer(&r, slice_words, (const uint64_t* (*)[NR_DPUS])&lanes, A_off);
            if (r.err) { fprintf(stderr, "push A0 failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
        }

        /* broadcast B */
        {
            const uint64_t* lanes[NR_DPUS];
            for (uint32_t i = 0; i < nr_of_dpus; i++) lanes[i] = (const uint64_t*)B;
            for (uint32_t i = nr_of_dpus; i < NR_DPUS; i++) lanes[i] = lanes[0];

            vud_simple_transfer(&r, vec_words, (const uint64_t* (*)[NR_DPUS])&lanes, B_off);
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
	// resume DPU
	vud_rank_rel_mux(&r);
    	if (r.err) { fprintf(stderr, "vud_rank_rel_mux failed\n"); return -1; }

        vud_ime_wait(&r);
        if (r.err) { fprintf(stderr, "wait failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

        if (rep >= p.n_warmup) {
            stop(&timer, 2);
#if ENERGY
            DPU_ASSERT(dpu_probe_stop(&probe));
#endif
        }

        /* intermediate layers */
        for (int lay = 1; lay < NUM_LAYERS; lay++) {
            c.cmd=CMD_RUN; c.job_id=1; c.status=ST_WAITING; c._pad=0;
	    push_ctrl_broadcast(&r, c);

            if (rep >= p.n_warmup) start(&timer, 4, rep - p.n_warmup);

            /* gather C_dpu */
            {
                uint64_t* lanes[NR_DPUS];
                for (uint32_t i = 0; i < nr_of_dpus; i++) {
                    lanes[i] = (uint64_t*)(C_dpu + (size_t)i * max_rows_per_dpu);
                }
                for (uint32_t i = nr_of_dpus; i < NR_DPUS; i++) lanes[i] = lanes[0];

                vud_simple_gather(&r, c_words, C_off, (uint64_t* (*)[NR_DPUS])&lanes);
                if (r.err) { fprintf(stderr, "gather C failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
            }

            /* B_tmp = C (packed) */
            {
                unsigned int idx = 0;
                for (unsigned int n = 0; n < nr_of_dpus; n++) {
                    for (unsigned int j = 0; j < dpu_info[n].rows_per_dpu; j++) {
                        B_tmp[idx] = C_dpu[n * max_rows_per_dpu + j];
                        idx++;
                    }
                }
                /* if n_size_pad > m_size, pad zeros for broadcast safety */
                for (unsigned int k = idx; k < n_size_pad; k++) B_tmp[k] = 0;
            }

            /* broadcast B_tmp */
            {
                const uint64_t* lanes[NR_DPUS];
                for (uint32_t i = 0; i < nr_of_dpus; i++) lanes[i] = (const uint64_t*)B_tmp;
                for (uint32_t i = nr_of_dpus; i < NR_DPUS; i++) lanes[i] = lanes[0];

                vud_simple_transfer(&r, vec_words, (const uint64_t* (*)[NR_DPUS])&lanes, B_off);
                if (r.err) { fprintf(stderr, "push B_tmp failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
            }

            /* scatter A[lay] slices */
            {
                const uint64_t* lanes[NR_DPUS];
                for (uint32_t i = 0; i < nr_of_dpus; i++) {
                    lanes[i] = (const uint64_t*)(A[lay] + (size_t)dpu_info[i].prev_rows_dpu * n_size);
                }
                for (uint32_t i = nr_of_dpus; i < NR_DPUS; i++) lanes[i] = lanes[0];

                vud_simple_transfer(&r, slice_words, (const uint64_t* (*)[NR_DPUS])&lanes, A_off);
                if (r.err) { fprintf(stderr, "push A[%d] failed: %s\n", lay, vud_error_str(r.err)); return EXIT_FAILURE; }
            }

            if (rep >= p.n_warmup) stop(&timer, 4);

            /* launch again */
            if (rep >= p.n_warmup) start(&timer, 2, rep - p.n_warmup);

	    vud_rank_rel_mux(&r);
    	    if (r.err) { fprintf(stderr, "vud_rank_rel_mux failed\n"); return -1; }

            vud_ime_wait(&r);
            if (r.err) { fprintf(stderr, "wait failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

            if (rep >= p.n_warmup) stop(&timer, 2);

        }
        c.cmd=CMD_EXIT; c.job_id=1; c.status=ST_WAITING; c._pad=0;
        push_ctrl_broadcast(&r, c);
	vud_rank_rel_mux(&r);
    	if (r.err) { fprintf(stderr, "vud_rank_rel_mux failed\n"); return -1; }
        vud_ime_wait(&r);
        if (r.err) { fprintf(stderr, "vud_ime_wait failed %d\n", r.err); return -1; }

        /* final gather for this rep */
        if (rep >= p.n_warmup) start(&timer, 3, rep - p.n_warmup);
        {
            uint64_t* lanes[NR_DPUS];
            for (uint32_t i = 0; i < nr_of_dpus; i++) {
                lanes[i] = (uint64_t*)(C_dpu + (size_t)i * max_rows_per_dpu);
            }
            for (uint32_t i = nr_of_dpus; i < NR_DPUS; i++) lanes[i] = lanes[0];

            vud_simple_gather(&r, c_words, C_off, (uint64_t* (*)[NR_DPUS])&lanes);
            if (r.err) { fprintf(stderr, "final gather C failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
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
	printf("Inter-DPU Time (ms): ");
	print(&timer, 4, p.n_reps);
	printf("DPU-CPU Time (ms): ");
	print(&timer, 3, p.n_reps);

        // update CSV
#define TEST_NAME "MLP"
#define RESULTS_FILE "prim_results.csv"
        //update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 0, p.n_reps, "CPU");
        update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 1, p.n_reps, "M_C2D");
        update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 3, p.n_reps, "M_D2C");
        update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 2, p.n_reps, "DPU");

#if ENERGY
	printf("Energy (J): %f J\t", avg_energy);
#endif
	printf("\n\n");

    // Check output
    bool status = true;
    unsigned int idx = 0;
    for (unsigned int n = 0; n < nr_of_dpus; n++) {
        for (unsigned int j = 0; j < dpu_info[n].rows_per_dpu; j++) {
            if (C[idx] != C_dpu[n * max_rows_per_dpu + j]) {
                status = false;
#if PRINT
                printf("%u: %u -- %u\n", idx, (unsigned)C[idx], (unsigned)C_dpu[n * max_rows_per_dpu + j]);
#endif
            }
            idx++;
        }
    }

    if (status) printf("\n[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    else        printf("\n[" ANSI_COLOR_RED   "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");

	// Deallocation
	for(int i = 0; i < NUM_LAYERS; i++)
		free(A[i]);
	free(A);
	free(B);
	free(C);
	free(C_dpu);

#if ENERGY
	DPU_ASSERT(dpu_probe_deinit(&probe));
#endif
    
        vud_rank_free(&r);
	return 0;
}
