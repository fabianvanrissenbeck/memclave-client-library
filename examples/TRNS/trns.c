/**
* app.c
* TRNS Host Application Source File
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <math.h>

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

#ifndef DPU_BINARY
#define DPU_BINARY "../trns"
#endif

#ifndef ARG_OFFSET
#define ARG_OFFSET   0x2000u
#endif
#define ARG_SIZE     ((uint32_t)sizeof(dpu_arguments_t))
#define A_OFFSET     (ARG_OFFSET + ((ARG_SIZE + 0xFFu) & ~0xFFu))

#if ENERGY
#include <dpu_probe.h>
#endif

// Pointer declaration
static T* A_host;
static T* A_backup;
static T* A_result;

// Create input arrays
static void read_input(T* A, unsigned int nr_elements) {
    srand(0);
    printf("nr_elements\t%u\t", nr_elements);
    for (unsigned int i = 0; i < nr_elements; i++) {
        A[i] = (T) (rand());
    }
}

static void push_args_array(vud_rank *r, const dpu_arguments_t *args, uint32_t nr_of_dpus) {
    const vud_mram_size words = (vud_mram_size)((sizeof(dpu_arguments_t) + 7u) / 8u);
    _Alignas(8) uint64_t staged[NR_DPUS][4];  // up to 32B args
    assert(nr_of_dpus <= NR_DPUS && words <= 4);

    for (uint32_t i = 0; i < nr_of_dpus; ++i) {
        memset(staged[i], 0, words * 8u);
        memcpy(staged[i], &args[i], sizeof(dpu_arguments_t));
    }
    const uint64_t *lanes[NR_DPUS];
    for (uint32_t i = 0; i < nr_of_dpus; ++i) lanes[i] = staged[i];
    for (uint32_t i = nr_of_dpus; i < NR_DPUS; ++i) lanes[i] = staged[0];
    vud_simple_transfer(r, words, (const uint64_t (*)[NR_DPUS])&lanes, ARG_OFFSET);
}

// Compute output in the host
static void trns_host(T* input, unsigned int A, unsigned int B, unsigned int b){
   T* output = (T*) malloc(sizeof(T) * A * B * b);
   unsigned int next;
   for (unsigned int j = 0; j < b; j++){
      for (unsigned int i = 0; i < A * B; i++){
         next = (i * A) - (A * B - 1) * (i / B);
         output[next * b + j] = input[i*b+j];
      }
   }
   for (unsigned int k = 0; k < A * B * b; k++){
      input[k] = output[k];
   }
   free(output);
}

// Main of the Host Application
int main(int argc, char **argv) {

    struct Params p = input_params(argc, argv);

    vud_rank r = vud_rank_alloc(VUD_ALLOC_ANY);
    if (r.err) { 
	    fprintf(stderr, "rank_alloc failed: %s\n", vud_error_str(r.err)); 
	    return EXIT_FAILURE; 
    }
    vud_ime_wait(&r);
    if (r.err) { 
	    fprintf(stderr, "ime_wait failed: %s\n", vud_error_str(r.err)); 
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
    
#if ENERGY
    struct dpu_probe_t probe;
    DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif

    unsigned int i = 0;
    unsigned int N_ = p.N_;
    const unsigned int n = p.n;
    const unsigned int M_ = p.M_;
    const unsigned int m = p.m;
    N_ = p.exp == 0 ? N_ * NR_DPUS : N_;

    // Input/output allocation
    A_host = malloc(M_ * m * N_ * n * sizeof(T));
    A_backup = malloc(M_ * m * N_ * n * sizeof(T));
    A_result = malloc(M_ * m * N_ * n * sizeof(T));
    T* done_host = malloc(M_ * n); // Host array to reset done array of step 3
    memset(done_host, 0, M_ * n);

    // Create an input file with arbitrary data
    read_input(A_host, M_ * m * N_ * n);
    memcpy(A_backup, A_host, M_ * m * N_ * n * sizeof(T));

    // Timer declaration
    Timer timer;

    printf("NR_TASKLETS\t%d\n", NR_TASKLETS);
    printf("M_\t%u, m\t%u, N_\t%u, n\t%u\n", M_, m, N_, n);

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        int timer_fix = 0;
        // Compute output on CPU (performance comparison and verification purposes)
        memcpy(A_host, A_backup, M_ * m * N_ * n * sizeof(T));
        if(rep >= p.n_warmup)
            start(&timer, 0, rep - p.n_warmup + timer_fix);
        trns_host(A_host, M_ * m, N_ * n, 1);
        if(rep >= p.n_warmup)
            stop(&timer, 0);

        unsigned int curr_dpu = 0;
        unsigned int active_dpus;
        unsigned int active_dpus_before = 0;
        unsigned int first_round = 1;

        for (unsigned base_col = 0; base_col < N_; base_col += NR_DPUS) {
            const unsigned active = (N_ - base_col < NR_DPUS) ? (N_ - base_col) : NR_DPUS;

            dpu_arguments_t args0[NR_DPUS];
            for (unsigned i = 0; i < active; i++) {
                args0[i].kernel = 0;
                args0[i].M_ = M_; args0[i].m = m; args0[i].n = n;
            }
            /* replicate to fill the lanes array safely */
            for (unsigned i = active; i < NR_DPUS; i++) 
	    args0[i] = args0[0];
            push_args_array(&r, args0, NR_DPUS);
            if (r.err) { 
		    fprintf(stderr, "args0 transfer failed: %s\n", vud_error_str(r.err)); 
		    return EXIT_FAILURE; 
	    }

            /* Push columns for these DPUs: row-by-row, n elements per DPU */
            const vud_mram_size words_row = (vud_mram_size)(((size_t)n * sizeof(T) + 7) / 8);
            const size_t rows = (size_t)M_ * m;
            printf("Load input data (step 1)\n");
            if(rep >= p.n_warmup)
                start(&timer, 1, rep - p.n_warmup + timer_fix);
            for (size_t row = 0; row < rows; row++) {
                const uint64_t *lanes[NR_DPUS];
                for (unsigned i = 0; i < active; i++)
                    lanes[i] = (const uint64_t*)(A_backup + row * (size_t)N_ * n + (size_t)(base_col + i) * n);
                for (unsigned i = active; i < NR_DPUS; i++) 
			lanes[i] = lanes[0];
                vud_simple_transfer(&r, words_row, (const uint64_t (*)[NR_DPUS])&lanes,
                                    (vud_mram_addr)(A_OFFSET + row * (size_t)n * sizeof(T)));
                if (r.err) { 
			fprintf(stderr, "push A(row=%zu) failed: %s\n", row, vud_error_str(r.err)); 
			return EXIT_FAILURE; 
		}
            }
            if(rep >= p.n_warmup)
                stop(&timer, 1);
            printf("Run step 2 on DPU(s) \n");
            // Run DPU kernel
            if(rep >= p.n_warmup){
                start(&timer, 2, rep - p.n_warmup + timer_fix);
#if ENERGY
                DPU_ASSERT(dpu_probe_start(&probe));
#endif
            }
            vud_ime_launch(&r);
            if (r.err) { 
		    fprintf(stderr, "launch step2 failed: %s\n", vud_error_str(r.err)); 
		    return EXIT_FAILURE; 
	    }
            vud_ime_wait(&r);
            if (r.err) { 
		    fprintf(stderr, "wait step2 failed: %s\n", vud_error_str(r.err)); 
		    return EXIT_FAILURE; 
	    }
            if(rep >= p.n_warmup){
                stop(&timer, 2);
#if ENERGY
                DPU_ASSERT(dpu_probe_stop(&probe));
#endif
            }
	vud_rank_rel_mux(&r);

	vud_ime_wait(&r);
#if PRINT
        {
            unsigned int each_dpu = 0;
            printf("Display DPU Logs\n");
            DPU_FOREACH (dpu_set, dpu) {
                printf("DPU#%d:\n", each_dpu);
                DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
                each_dpu++;
            }
        }
#endif
            memset(done_host, 0, (size_t)M_ * n);
            {
                const vud_mram_size words_done = (vud_mram_size)(((size_t)M_ * n + 7) / 8);
                const uint64_t *lanes[NR_DPUS];
                for (unsigned i = 0; i < active; i++) 
			lanes[i] = (const uint64_t*)done_host;
                for (unsigned i = active; i < NR_DPUS; i++) 
			lanes[i] = lanes[0];
                const vud_mram_addr done_off = (vud_mram_addr)(A_OFFSET + (vud_mram_addr)((size_t)M_ * m * n * sizeof(T)));
                vud_simple_transfer(&r, words_done, (const uint64_t (*)[NR_DPUS])&lanes, done_off);
                if (r.err) { 
			fprintf(stderr, "push done[] failed: %s\n", vud_error_str(r.err)); 
			return EXIT_FAILURE; 
		}
            }

            dpu_arguments_t args1[NR_DPUS];
            for (unsigned i = 0; i < active; i++) {
                args1[i].kernel = 1;
                args1[i].M_ = M_; args1[i].m = m; args1[i].n = n;
            }
            for (unsigned i = active; i < NR_DPUS; i++) 
		    args1[i] = args1[0];
            push_args_array(&r, args1, NR_DPUS);
            if (r.err) { 
		    fprintf(stderr, "args1 transfer failed: %s\n", vud_error_str(r.err)); 
		    return EXIT_FAILURE; 
	    }
            printf("Run step 3 on DPU(s) \n");
            // Run DPU kernel
            if(rep >= p.n_warmup){
                start(&timer, 3, rep - p.n_warmup + timer_fix);
#if ENERGY
                DPU_ASSERT(dpu_probe_start(&probe));
#endif
            }
            vud_ime_launch(&r);
            if (r.err) { 
		    fprintf(stderr, "launch step3 failed: %s\n", vud_error_str(r.err)); 
		    return EXIT_FAILURE; 
	    }
            vud_ime_wait(&r);
            if (r.err) { 
		    fprintf(stderr, "wait step3 failed: %s\n", vud_error_str(r.err)); 
		    return EXIT_FAILURE; 
	    }
            if(rep >= p.n_warmup){
                stop(&timer, 3);
#if ENERGY
                DPU_ASSERT(dpu_probe_stop(&probe));
#endif
            }
	vud_rank_rel_mux(&r);

	vud_ime_wait(&r);
#if PRINT
        {
            unsigned int each_dpu = 0;
            printf("Display DPU Logs\n");
            DPU_FOREACH (dpu_set, dpu) {
                printf("DPU#%d:\n", each_dpu);
                DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
                each_dpu++;
            }
        }
#endif

            printf("Retrieve results\n");
            if(rep >= p.n_warmup)
                start(&timer, 4, rep - p.n_warmup + timer_fix);
            /* Gather full DPU slab*/
            const size_t slab_elems  = (size_t)M_ * m * n;           // elements per DPU
            const size_t slab_bytes  = slab_elems * sizeof(T);
            const vud_mram_size wslab = (vud_mram_size)((slab_bytes + 7) / 8);
           
            uint64_t *dsts[NR_DPUS];
            for (unsigned i = 0; i < active; i++) {
                dsts[i] = (uint64_t*)(A_result + (size_t)(base_col + i) * slab_elems);
            }
            for (unsigned i = active; i < NR_DPUS; i++) dsts[i] = dsts[0];
           
            vud_simple_gather(&r, wslab, (vud_mram_addr)A_OFFSET, &dsts);
            if (r.err) {
                fprintf(stderr, "gather slab failed: %s\n", vud_error_str(r.err));
                return EXIT_FAILURE;
            }

            if(rep >= p.n_warmup)
                stop(&timer, 4);

            timer_fix++;
        }
    }

    // Print timing results
    printf("CPU ");
    print(&timer, 0, p.n_reps);
    printf("CPU-DPU (Step 1) ");
    print(&timer, 1, p.n_reps);
    printf("Step 2 ");
    print(&timer, 2, p.n_reps);
    printf("Step 3 ");
    print(&timer, 3, p.n_reps);
    printf("DPU-CPU ");
    print(&timer, 4, p.n_reps);

    // update CSV
#define TEST_NAME "TRNS"
#define RESULTS_FILE "prim_results.csv"
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 0, p.n_reps, "CPU");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 1, p.n_reps, "M_C2D");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 4, p.n_reps, "M_D2C");
    double dpu_ms = prim_timer_ms_avg(&timer, 2, p.n_reps) + prim_timer_ms_avg(&timer, 3, p.n_reps);
    update_csv(RESULTS_FILE, TEST_NAME, "DPU", dpu_ms);

    #if ENERGY
    double energy;
    DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &energy));
    printf("DPU Energy (J): %f\t", energy);
    #endif	

    // Check output
    bool status = true;
    for (i = 0; i < M_ * m * N_ * n; i++) {
        if(A_host[i] != A_result[i]){ 
            status = false;
#if PRINT
            printf("%d: %lu -- %lu\n", i, A_host[i], A_result[i]);
#endif
        }
    }
    if (status) {
        printf("\n[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("\n[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }

    // Deallocation
    free(A_host);
    free(A_backup);
    free(A_result);
    free(done_host);
    vud_rank_free(&r);
	
    return status ? 0 : -1;
}
