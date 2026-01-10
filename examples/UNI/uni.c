/**
* app.c
* UNI Host Application Source File
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>

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
#define DPU_BINARY "../uni"
#endif

// MRAM layout
#ifndef ARG_OFFSET
#define ARG_OFFSET   0x2000u
#endif
#define ARG_SIZE     ((uint32_t)sizeof(dpu_arguments_t))
#define A_OFFSET     (ARG_OFFSET + ((ARG_SIZE + 0xFFu) & ~0xFFu))


#if ENERGY
#include <dpu_probe.h>
#endif

// Pointer declaration
static T* A;
static T* C;
static T* C2;

// Create input arrays
static void read_input(T* A, unsigned int nr_elements, unsigned int nr_elements_round) {
    //srand(0);
    printf("nr_elements\t%u\t", nr_elements);
    for (unsigned int i = 0; i < nr_elements; i++) {
        //A[i] = (T) (rand());
        A[i] = i%2==0?i:i+1;
    }
    for (unsigned int i = nr_elements; i < nr_elements_round; i++) {
        A[i] = A[nr_elements - 1];
    }
}

// Compute output in the host
static unsigned int unique_host(T* C, T* A, unsigned int nr_elements) {
    unsigned int pos = 0;
    C[pos] = A[pos];
    pos++;
    for(unsigned int i = 1; i < nr_elements; i++) {
        if(A[i] != A[i-1]) {
            C[pos] = A[i];
            pos++;
        }
    }
    return pos;
}

static void push_args_array(vud_rank *r, dpu_arguments_t *args, uint32_t nr_of_dpus) {
    const vud_mram_size words = (vud_mram_size)((sizeof(dpu_arguments_t) + 7u) / 8u);
    _Alignas(8) uint64_t staged[NR_DPUS][4]; // up to 32B args
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

    const uint32_t nr_of_dpus = NR_DPUS;
    printf("Allocated %d DPU(s)\n", nr_of_dpus);

    unsigned int i = 0;
    uint32_t acc = 0;
    uint32_t total_count = 0;

    const unsigned int input_size = p.exp == 0 ? p.input_size * nr_of_dpus : p.input_size; // Total input size (weak or strong scaling)
    const unsigned int input_size_dpu_ = divceil(input_size, nr_of_dpus); // Input size per DPU (max.)
    const unsigned int input_size_dpu_round = 
        (input_size_dpu_ % (NR_TASKLETS * REGS) != 0) ? roundup(input_size_dpu_, (NR_TASKLETS * REGS)) : input_size_dpu_; // Input size per DPU (max.), 8-byte aligned
												   const size_t bytes_per_dpu = (size_t)input_size_dpu_round * sizeof(T);

    // Input/output allocation
    A = malloc(input_size_dpu_round * nr_of_dpus * sizeof(T));
    C = malloc(input_size_dpu_round * nr_of_dpus * sizeof(T));
    C2 = malloc(input_size_dpu_round * nr_of_dpus * sizeof(T));
    T *bufferA = A;
    T *bufferC = C2;

    // Create an input file with arbitrary data
    read_input(A, input_size, input_size_dpu_round * nr_of_dpus);

    // Timer declaration
    Timer timer;

    printf("NR_TASKLETS\t%d\tBL\t%d\n", NR_TASKLETS, BL);

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Compute output on CPU
        if(rep >= p.n_warmup)
            start(&timer, 0, rep - p.n_warmup);
        total_count = unique_host(C, A, input_size);
        if(rep >= p.n_warmup)
            stop(&timer, 0);

        if(rep >= p.n_warmup)
            start(&timer, 1, rep - p.n_warmup);
        // Input arguments
        dpu_arguments_t args[NR_DPUS];
        for (uint32_t i = 0; i < nr_of_dpus; ++i) {
            args[i].size   = (uint32_t)bytes_per_dpu;
            args[i].kernel = kernel1; // single kernel
        }
        push_args_array(&r, args, nr_of_dpus);
        if (r.err) { fprintf(stderr, "args transfer failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

        {
            const vud_mram_size wordsA = (vud_mram_size)((bytes_per_dpu + 7) / 8);
            const uint64_t *lanes[NR_DPUS];
            for (uint32_t i = 0; i < nr_of_dpus; ++i)
                lanes[i] = (const uint64_t*)(A + (size_t)i * input_size_dpu_round);
            for (uint32_t i = nr_of_dpus; i < NR_DPUS; ++i) lanes[i] = lanes[0];
            vud_simple_transfer(&r, wordsA, (const uint64_t (*)[NR_DPUS])&lanes, A_OFFSET);
            if (r.err) { fprintf(stderr, "push A failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
        }
        if(rep >= p.n_warmup)
            stop(&timer, 1);

        // Run DPU kernel
        if(rep >= p.n_warmup) {
            start(&timer, 2, rep - p.n_warmup);
            #if ENERGY
            DPU_ASSERT(dpu_probe_start(&probe));
            #endif
        }
        vud_ime_launch(&r);
        if (r.err) { 
		fprintf(stderr, "launch failed: %s\n", vud_error_str(r.err)); 
		return EXIT_FAILURE; 
	}
        vud_ime_wait(&r);
        if (r.err) { 
		fprintf(stderr, "wait failed: %s\n", vud_error_str(r.err)); 
		return EXIT_FAILURE; 
	}
        if(rep >= p.n_warmup) {
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

        dpu_results_t results[nr_of_dpus];
        uint32_t* results_scan = malloc(nr_of_dpus * sizeof(uint32_t));
        uint32_t* offset = calloc(nr_of_dpus, sizeof(uint32_t));
        uint32_t* offset_scan = calloc(nr_of_dpus, sizeof(uint32_t));
        i = 0;
        acc = 0;

        if(rep >= p.n_warmup)
            start(&timer, 3, rep - p.n_warmup);
        // RETRIEVE TRANSFER
        uint64_t per_dpu_counts_64[NR_DPUS];
        {
            uint64_t *ptrs[NR_DPUS];
            for (uint32_t d = 0; d < nr_of_dpus; ++d) ptrs[d] = &per_dpu_counts_64[d];
            for (uint32_t d = nr_of_dpus; d < NR_DPUS; ++d) ptrs[d] = ptrs[0];
            vud_simple_gather(&r, 1, SK_LOG_OFFSET, &ptrs);
            if (r.err) { fprintf(stderr, "gather counts failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
        }
        // Convert + exclusive scan for global placement
        T per_dpu_counts[NR_DPUS];
        T per_dpu_offsets[NR_DPUS];
        for (uint32_t d = 0; d < nr_of_dpus; ++d) {
            per_dpu_counts[d] = (T)per_dpu_counts_64[d];
            per_dpu_offsets[d] = acc;
            acc += per_dpu_counts[d];
        }
        const size_t total_out = (size_t)acc;
        if(rep >= p.n_warmup)
		    stop(&timer, 3);

        i = 0;
        if(rep >= p.n_warmup)
            start(&timer, 4, rep - p.n_warmup);
        const vud_mram_addr B_base  = (vud_mram_addr)(A_OFFSET + (vud_mram_addr)bytes_per_dpu);
        const vud_mram_size wordsB  = (vud_mram_size)((bytes_per_dpu + 7) / 8);

        // Scratch to gather per-DPU B fully (fixed width), then splice compactly
        T *scratch = malloc((size_t)input_size_dpu_round * nr_of_dpus * sizeof(T));
        if (!scratch) { fprintf(stderr, "scratch malloc failed\n"); return EXIT_FAILURE; }

        {
            uint64_t *dsts[NR_DPUS];
            for (uint32_t i = 0; i < nr_of_dpus; ++i)
                dsts[i] = (uint64_t*)(scratch + (size_t)i * input_size_dpu_round);
            for (uint32_t i = nr_of_dpus; i < NR_DPUS; ++i) dsts[i] = dsts[0];
            vud_simple_gather(&r, wordsB, B_base, &dsts);
            if (r.err) { fprintf(stderr, "gather B failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
        }
        // Stitch compactly
        size_t w = 0;                 // write cursor into C2
        int have_last = 0;
        T last = 0;

        for (uint32_t i = 0; i < nr_of_dpus; ++i) {
            size_t cnt = (size_t)per_dpu_counts[i];
            if (!cnt) continue;

            T *src = scratch + (size_t)i * input_size_dpu_round;
            size_t start = 0;

            // If first of this DPU equals the globally last written value, skip it
            if (have_last && src[0] == last) {
                start = 1;
                if (cnt) cnt -= 1;
            }
            if (cnt) {
                memcpy(C2 + w, src + start, cnt * sizeof(T));
                w   += cnt;
                last = src[start + cnt - 1];
                have_last = 1;
            }
        }
        acc = (uint32_t)w; 
        free(scratch);
        if(rep >= p.n_warmup)
            stop(&timer, 4);

    }

    // Print timing results
    printf("CPU ");
    print(&timer, 0, p.n_reps);
    printf("CPU-DPU ");
    print(&timer, 1, p.n_reps);
    printf("DPU Kernel ");
    print(&timer, 2, p.n_reps);
    printf("Inter-DPU ");
    print(&timer, 3, p.n_reps);
    printf("DPU-CPU ");
    print(&timer, 4, p.n_reps);

    // update CSV
#define TEST_NAME "UNI"
#define RESULTS_FILE "prim_results.csv"
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 1, p.n_reps, "M_C2D");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 4, p.n_reps, "M_D2C");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 2, p.n_reps, "DPU");

#if ENERGY
    double energy;
    DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &energy));
    printf("DPU Energy (J): %f\t", energy);
#endif	

    // Check output
    bool status = true;
    if(acc != total_count) status = false;
    //printf("acc %u, total_count %u\n", acc, total_count);
    for (i = 0; i < acc; i++) {
        if(C[i] != bufferC[i]){ 
            status = false;
#if PRINT
            printf("%d: %lu -- %lu\n", i, C[i], bufferC[i]);
#endif
        }
    }
    if (status) {
        printf("\n[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("\n[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }

    // Deallocation
    free(A);
    free(C);
    free(C2);
    vud_rank_free(&r);
    return status ? 0 : -1;
}
