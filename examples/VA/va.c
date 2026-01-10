/**
* app.c
* VA Host Application Source File
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

#define NR_DPUS     64
#define NR_TASKLETS 16

#include <vud.h>
#include <vud_sk.h>
#include <vud_ime.h>
#include <vud_mem.h>
#include "../../src/vud_log.h"
#include "support/common.h"
#include "support/timer.h"
#include "support/params.h"
#include "support/prim_results.h"

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "../va"
#endif

#ifndef ARG_OFFSET
#define ARG_OFFSET   0x2000u
#endif
#define ARG_SIZE     ((uint32_t)sizeof(dpu_arguments_t))
#define A_OFFSET     (ARG_OFFSET + ((ARG_SIZE + 0xFFu) & ~0xFFu))

#ifndef SK_LOG_OFFSET
#define SK_LOG_OFFSET ((64u << 20) - 64u)
#endif

#define LOG_WORDS 8
#define LOG_MAGIC 0x534B4C4F475631ULL /* "SKLOGV1" */

#if ENERGY
#include <dpu_probe.h>
#endif

// Pointer declaration
static T* A;
static T* B;
static T* C_ref;
static T* C_dpu;

// Create input arrays
static void read_input(T* A, T* B, unsigned int nr_elements) {
    srand(0);
    printf("nr_elements\t%u\t", nr_elements);
    for (unsigned int i = 0; i < nr_elements; i++) {
        A[i] = (T) (rand());
        B[i] = (T) (rand());
    }
}

// Compute output in the host
static void vector_addition_host(T* C, T* A, T* B, unsigned int nr_elements) {
    for (unsigned int i = 0; i < nr_elements; i++) {
        C[i] = A[i] + B[i];
    }
}

// push dpu_arguments_t array to all DPUs via pointer-array scatter
static void push_args_array(vud_rank *r, dpu_arguments_t *args, uint32_t nr_of_dpus) {
    const vud_mram_size words = (vud_mram_size)((sizeof(dpu_arguments_t) + 7u) / 8u);
    _Alignas(8) uint64_t staged[NR_DPUS][4];  // up to 32B args; bump if needed
    assert(nr_of_dpus <= NR_DPUS);
    assert(words <= 4);

    for (uint32_t i = 0; i < nr_of_dpus; ++i) {
        memset(staged[i], 0, words * 8u);
        memcpy(staged[i], &args[i], sizeof(dpu_arguments_t));
    }
    const uint64_t *lanes[NR_DPUS];
    for (uint32_t i = 0; i < nr_of_dpus; ++i) lanes[i] = staged[i];
    for (uint32_t i = nr_of_dpus; i < NR_DPUS; ++i) lanes[i] = staged[0];

    vud_simple_transfer(r, words, (const uint64_t (*)[NR_DPUS])&lanes, ARG_OFFSET);
}


// Main of the Host Application
int main(int argc, char **argv) {

    struct Params p = input_params(argc, argv);

    vud_rank r = vud_rank_alloc(VUD_ALLOC_ANY);
    if (r.err) { fprintf(stderr, "rank_alloc failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

    vud_ime_wait(&r);
    if (r.err) { fprintf(stderr, "ime_wait failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

    vud_ime_load(&r, DPU_BINARY);
    if (r.err) { fprintf(stderr, "cannot load subkernel: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

    vud_rank_nr_workers(&r, 12);
    if (r.err) { fprintf(stderr, "cannot start worker threads: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

    uint32_t nr_of_dpus = NR_DPUS;
    unsigned int i = 0;

    uint8_t key[32];
    //random_key(key);
 
    //vud_ime_install_key(&r, key, NULL, NULL);
 
    if (r.err) { puts("key exchange failed"); return -1; }

#if ENERGY
    struct dpu_probe_t probe;
    DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif

    // Allocate DPUs and load binary

    const unsigned int input_size = (p.exp == 0) ? (p.input_size * nr_of_dpus) : p.input_size;

    const unsigned int input_size_8bytes = ((input_size * sizeof(T)) % 8) != 0 ? roundup(input_size, 8) : input_size;

    const unsigned int input_size_dpu = divceil(input_size, nr_of_dpus);
    const unsigned int input_size_dpu_8bytes = ((input_size_dpu * sizeof(T)) % 8) != 0 ? roundup(input_size_dpu, 8) : input_size_dpu;

    /* host buffers (same shape as PRiM: per-DPU slices padded to xfer size) */
    A     = (T*)malloc((size_t)input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    B     = (T*)malloc((size_t)input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    C_ref = (T*)malloc((size_t)input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    C_dpu = (T*)malloc((size_t)input_size_dpu_8bytes * nr_of_dpus * sizeof(T));

    T *bufferA = A;
    T *bufferB = B;
    T *bufferC = C_dpu;

    // Create an input file with arbitrary data
    read_input(A, B, input_size);
    const vud_mram_addr A_off = (vud_mram_addr)A_OFFSET;
    const vud_mram_addr B_off = (vud_mram_addr)(A_OFFSET + (vud_mram_addr)(input_size_dpu_8bytes * sizeof(T)));

    /* Safety: A + B regions should not collide with log region */
    const uint64_t xfer_bytes = (uint64_t)input_size_dpu_8bytes * (uint64_t)sizeof(T);
    assert((uint64_t)B_off + xfer_bytes <= (uint64_t)SK_LOG_OFFSET);


    // Timer declaration
    Timer timer;

    printf("NR_DPU\t%d\tBL\t%d\n", NR_DPUS, BL);

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        /* CPU reference */
        if (rep >= p.n_warmup) start(&timer, 0, rep - p.n_warmup);
        vector_addition_host(C_ref, A, B, input_size);
        if (rep >= p.n_warmup) stop(&timer, 0);


        if (rep >= p.n_warmup) start(&timer, 1, rep - p.n_warmup);

        /* args per DPU */
        unsigned int kernel = 0;
        dpu_arguments_t input_arguments[NR_DPUS];

        for (uint32_t i = 0; i < nr_of_dpus - 1; i++) {
            input_arguments[i].size          = input_size_dpu_8bytes * sizeof(T);
            input_arguments[i].transfer_size = input_size_dpu_8bytes * sizeof(T);
            input_arguments[i].kernel        = kernel;
        }
        input_arguments[nr_of_dpus - 1].size =
            (input_size_8bytes - input_size_dpu_8bytes * (nr_of_dpus - 1)) * sizeof(T);
        input_arguments[nr_of_dpus - 1].transfer_size = input_size_dpu_8bytes * sizeof(T);
        input_arguments[nr_of_dpus - 1].kernel        = kernel;

        push_args_array(&r, input_arguments, nr_of_dpus);
        if (r.err) { fprintf(stderr, "push args failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

        /* scatter A and B into contiguous regions (A then B) */
        const vud_mram_size words = (vud_mram_size)((input_size_dpu_8bytes * sizeof(T)) / 8u);

        /* push A */
        {
            const uint64_t *lanes[NR_DPUS];
            for (uint32_t d = 0; d < nr_of_dpus; ++d)
                lanes[d] = (const uint64_t*)(bufferA + (size_t)input_size_dpu_8bytes * d);
            for (uint32_t d = nr_of_dpus; d < NR_DPUS; ++d) lanes[d] = lanes[0];

            vud_simple_transfer(&r, words, (const uint64_t (*)[NR_DPUS])&lanes, A_off);
            if (r.err) { fprintf(stderr, "push A failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
        }

        /* push B */
        {
            const uint64_t *lanes[NR_DPUS];
            for (uint32_t d = 0; d < nr_of_dpus; ++d)
                lanes[d] = (const uint64_t*)(bufferB + (size_t)input_size_dpu_8bytes * d);
            for (uint32_t d = nr_of_dpus; d < NR_DPUS; ++d) lanes[d] = lanes[0];

            vud_simple_transfer(&r, words, (const uint64_t (*)[NR_DPUS])&lanes, B_off);
            if (r.err) { fprintf(stderr, "push B failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
        }

        if (rep >= p.n_warmup) stop(&timer, 1);

        // 3) launch + wait
        if (rep >= p.n_warmup) start(&timer, 2, rep - p.n_warmup);
        vud_ime_launch(&r);
        if (r.err) { fprintf(stderr, "launch failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

        vud_ime_wait(&r);
        if (r.err) { fprintf(stderr, "wait failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

        if (rep >= p.n_warmup) stop(&timer, 2);
	vud_rank_rel_mux(&r);

	vud_ime_wait(&r);

        if (rep >= p.n_warmup) start(&timer, 3, rep - p.n_warmup);

        /* gather output from B region (PRiM semantics: B += A written back into B) */
        {
            uint64_t *lanes[NR_DPUS];
            for (uint32_t d = 0; d < nr_of_dpus; ++d)
                lanes[d] = (uint64_t*)(bufferC + (size_t)input_size_dpu_8bytes * d);
            for (uint32_t d = nr_of_dpus; d < NR_DPUS; ++d) lanes[d] = lanes[0];

            vud_simple_gather(&r, words, B_off, (uint64_t* (*)[NR_DPUS])&lanes);
            if (r.err) { fprintf(stderr, "gather C failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
        }

        if (rep >= p.n_warmup) stop(&timer, 3);
      }


       // Print timing results
       printf("CPU ");
       print(&timer, 0, p.n_reps);
       printf("CPU-DPU ");
       print(&timer, 1, p.n_reps);
       printf("DPU Kernel ");
       print(&timer, 2, p.n_reps);
       printf("DPU-CPU ");
       print(&timer, 3, p.n_reps);
    // update CSV
#define TEST_NAME "VA"
#define RESULTS_FILE "prim_results.csv"
    //update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 0, p.n_reps, "CPU");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 1, p.n_reps, "M_C2D");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 3, p.n_reps, "M_D2C");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 2, p.n_reps, "DPU");


    /* Check output */
    bool status = true;
    for (unsigned int i = 0; i < input_size; i++) {
        if (C_ref[i] != C_dpu[i]) {
            status = false;
#if PRINT
            printf("%u: %u -- %u\n", i, C_ref[i], C_dpu[i]);
#endif
            break;
        }
    }
    if (status) printf("\n[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    else        printf("\n[" ANSI_COLOR_RED   "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");

    free(A); free(B); free(C_ref); free(C_dpu);
    vud_rank_free(&r);
    return status ? 0 : -1;
}
