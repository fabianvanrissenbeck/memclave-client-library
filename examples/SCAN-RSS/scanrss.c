/**
* app.c
* SCAN-RSS Host Application Source File
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

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "../scanrss"
#endif

#ifndef ARG_OFFSET
#define ARG_OFFSET   0x2000u
#endif
#define ARG_SIZE     ((uint32_t)sizeof(dpu_arguments_t))
#define A_OFFSET     (ARG_OFFSET + ((ARG_SIZE + 0xFFu) & ~0xFFu))

/// total MRAM per DPU
#define MRAM_SIZE_BYTES     (64u << 20)
/// we reserve 64B at the very top
#define SK_LOG_SIZE_BYTES   64
#define SK_LOG_OFFSET       (MRAM_SIZE_BYTES - SK_LOG_SIZE_BYTES)


#if ENERGY
#include <dpu_probe.h>
#endif

//enum { kernel0 = 0, kernel1 = 1};

// Pointer declaration
static T* A;
static T* C;
static T* C2;

// Create input arrays
static void read_input(T* A, unsigned int nr_elements, unsigned int nr_elements_round) {
    srand(0);
    printf("nr_elements\t%u\t", nr_elements);
    for (unsigned int i = 0; i < nr_elements; i++) {
        A[i] = (T) (rand());
    }
    for (unsigned int i = nr_elements; i < nr_elements_round; i++) {
        A[i] = 0;
    }
}

// Compute output in the host
static void scan_host(T* C, T* A, unsigned int nr_elements) {
    C[0] = A[0];
    for (unsigned int i = 1; i < nr_elements; i++) {
        C[i] = C[i - 1] + A[i];
    }
}

static void push_args_array(vud_rank *r, dpu_arguments_t *args, uint32_t nr_of_dpus) {
    const vud_mram_size words = (vud_mram_size)((sizeof(dpu_arguments_t) + 7u) / 8u);
    _Alignas(8) uint64_t staged[NR_DPUS][4];  // up to 32B args
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
    if (r.err) { 
	    fprintf(stderr, "rank_alloc failed: %s\n", vud_error_str(r.err)); 
	    return EXIT_FAILURE; 
    }
    vud_ime_wait(&r);
    if (r.err) { 
	    fprintf(stderr, "ime_wait failed: %s\n", vud_error_str(r.err)); 
	    return EXIT_FAILURE; 
    }
    vud_rank_nr_workers(&r, 12);
    if (r.err) { 
	    fprintf(stderr, "cannot start worker threads: %s\n", vud_error_str(r.err)); 
	    return EXIT_FAILURE; 
    }
    vud_ime_load(&r, DPU_BINARY);
    if (r.err) { 
	    fprintf(stderr, "cannot load subkernel: %s\n", vud_error_str(r.err)); 
	    return EXIT_FAILURE; 
    }

    const uint32_t nr_of_dpus = NR_DPUS;    

#if ENERGY
    struct dpu_probe_t probe;
    DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif

    printf("Allocated %d DPU(s)\n", nr_of_dpus);

    unsigned int i = 0;
    T accum = 0;

    const unsigned int input_size = p.exp == 0 ? p.input_size * nr_of_dpus : p.input_size; // Total input size (weak or strong scaling)
    const unsigned int input_size_dpu_ = divceil(input_size, nr_of_dpus); // Input size per DPU (max.)
    const unsigned int input_size_dpu_round = 
        (input_size_dpu_ % (NR_TASKLETS * REGS) != 0) ? roundup(input_size_dpu_, (NR_TASKLETS * REGS)) : input_size_dpu_; // Input size per DPU (max.), 8-byte aligned
    const size_t   bytes_per_dpu= (size_t)input_size_dpu_round * sizeof(T);
    const size_t   bytes_total  = (size_t)input_size_dpu_round * sizeof(T) * nr_of_dpus;

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

        // Compute output on CPU (performance comparison and verification purposes)
        if(rep >= p.n_warmup)
            start(&timer, 0, rep - p.n_warmup);
        scan_host(C, A, input_size);
        if(rep >= p.n_warmup)
            stop(&timer, 0);

        printf("Load input data\n");
        if(rep >= p.n_warmup)
            start(&timer, 1, rep - p.n_warmup);
        // Input arguments
        dpu_arguments_t args0[NR_DPUS];
        for (uint32_t i = 0; i < nr_of_dpus; ++i) {
            args0[i].size   = (uint32_t)bytes_per_dpu;
            args0[i].kernel = 0;
            args0[i].t_count= (T)0;
        }
        push_args_array(&r, args0, nr_of_dpus);
        if (r.err) { 
		fprintf(stderr, "args0 transfer failed: %s\n", vud_error_str(r.err)); 
		return EXIT_FAILURE; 
	}

        // Push A
        {
            const vud_mram_size wordsA = (vud_mram_size)((bytes_per_dpu + 7) / 8);
            const uint64_t *lanes[NR_DPUS];
            for (uint32_t i = 0; i < nr_of_dpus; ++i) 
		    lanes[i] = (const uint64_t*)(A + (size_t)i * input_size_dpu_round);
            for (uint32_t i = nr_of_dpus; i < NR_DPUS; ++i) 
		    lanes[i] = lanes[0];
            vud_simple_transfer(&r, wordsA, (const uint64_t (*)[NR_DPUS])&lanes, A_OFFSET);
            if (r.err) { 
		    fprintf(stderr, "push A failed: %s\n", vud_error_str(r.err)); 
		    return EXIT_FAILURE; 
	    }
        }
        if(rep >= p.n_warmup)
            stop(&timer, 1);

        printf("Run program on DPU(s) \n");
        // Run DPU kernel
        if(rep >= p.n_warmup) {
            start(&timer, 2, rep - p.n_warmup);
            #if ENERGY
            DPU_ASSERT(dpu_probe_start(&probe));
            #endif
        }
 
	vud_ime_launch(&r);
        if (r.err) {
            fprintf(stderr, "launch k0 failed: %s\n", vud_error_str(r.err));
            return EXIT_FAILURE;
        }
        vud_ime_wait(&r);
        if (r.err) {
            fprintf(stderr, "wait k0 failed: %s\n", vud_error_str(r.err));
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

        printf("Retrieve results\n");
        //dpu_results_t results[nr_of_dpus];
        //T* results_scan = malloc(nr_of_dpus * sizeof(T));
        //i = 0;
        //accum = 0;
		
        if(rep >= p.n_warmup)
            start(&timer, 3, rep - p.n_warmup);
        uint64_t per_dpu_totals[NR_DPUS];
        {
            uint64_t *ptrs[NR_DPUS];
            for (uint32_t d = 0; d < nr_of_dpus; ++d) ptrs[d] = &per_dpu_totals[d];
            for (uint32_t d = nr_of_dpus; d < NR_DPUS; ++d) ptrs[d] = ptrs[0];
            vud_simple_gather(&r, /*words=*/1, /*offset=*/SK_LOG_OFFSET, &ptrs);
            if (r.err) { 
		    fprintf(stderr, "gather totals failed: %s\n", vud_error_str(r.err)); 
		    return EXIT_FAILURE; 
	    }
        }
        T per_dpu_offsets[NR_DPUS];
        {
            T acc = 0;
            for (uint32_t d = 0; d < nr_of_dpus; ++d) {
                per_dpu_offsets[d] = acc;
                acc += (T)per_dpu_totals[d];
            }
        }
        // Arguments for scan kernel (2nd kernel)
        dpu_arguments_t args1[NR_DPUS];
        for (uint32_t i = 0; i < nr_of_dpus; ++i) {
            args1[i].size   = (uint32_t)bytes_per_dpu;
            args1[i].kernel = 1;
            args1[i].t_count= per_dpu_offsets[i];
        }
        push_args_array(&r, args1, nr_of_dpus);
        if (r.err) { 
		fprintf(stderr, "args1 transfer failed: %s\n", vud_error_str(r.err)); 
		return EXIT_FAILURE; 
	}
        if(rep >= p.n_warmup)
            stop(&timer, 3);

        printf("Run program on DPU(s) \n");
        // Run DPU kernel
        if(rep >= p.n_warmup) {
            start(&timer, 4, rep - p.n_warmup);
            #if ENERGY
            DPU_ASSERT(dpu_probe_start(&probe));
            #endif
        }
        vud_ime_launch(&r);
        if (r.err) { 
		fprintf(stderr, "launch k1 failed: %s\n", vud_error_str(r.err)); 
		return EXIT_FAILURE; 
	}
        vud_ime_wait(&r);
        if (r.err) { 
		fprintf(stderr, "wait k1 failed: %s\n", vud_error_str(r.err)); 
		return EXIT_FAILURE; 
	}
        if(rep >= p.n_warmup) {
            stop(&timer, 4);
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
            start(&timer, 5, rep - p.n_warmup);
        if (rep >= p.n_warmup) start(&timer, 6, rep - p.n_warmup);
        {
            const vud_mram_addr B_base = (vud_mram_addr)(A_OFFSET + (vud_mram_addr)bytes_per_dpu);
            const vud_mram_size wordsB = (vud_mram_size)((bytes_per_dpu + 7) / 8);
            uint64_t *dsts[NR_DPUS];
            for (uint32_t i = 0; i < nr_of_dpus; ++i) dsts[i] = (uint64_t*)(C2 + (size_t)i * input_size_dpu_round);
            for (uint32_t i = nr_of_dpus; i < NR_DPUS; ++i) dsts[i] = dsts[0];
            vud_simple_gather(&r, wordsB, B_base, &dsts);
            if (r.err) { fprintf(stderr, "gather B failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
        }
        if(rep >= p.n_warmup)
            stop(&timer, 5);

        // Free memory
        //free(results_scan);
    }

    // Print timing results
    printf("CPU ");
    print(&timer, 0, p.n_reps);
    printf("CPU-DPU ");
    print(&timer, 1, p.n_reps);
    printf("DPU Kernel Reduction ");
    print(&timer, 2, p.n_reps);
    printf("Inter-DPU (Scan) ");
    print(&timer, 3, p.n_reps);
    printf("DPU Kernel Scan ");
    print(&timer, 4, p.n_reps);
    printf("DPU-CPU ");
    print(&timer, 5, p.n_reps);

    // update CSV
#define TEST_NAME "SCAN-RSS"
#define RESULTS_FILE "prim_results.csv"
    //update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 0, p.n_reps, "CPU");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 1, p.n_reps, "M_C2D");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 5, p.n_reps, "M_D2C");
    double dpu_ms = prim_timer_ms_avg(&timer, 2, p.n_reps) + prim_timer_ms_avg(&timer, 4, p.n_reps);
    update_csv(RESULTS_FILE, TEST_NAME, "DPU", dpu_ms);

    #if ENERGY
    double energy;
    DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &energy));
    printf("DPU Energy (J): %f\t", energy);
    #endif	


    // Check output
    bool status = true;
    for (i = 0; i < input_size; i++) {
        if(C[i] != bufferC[i]){ 
            status = false;
#if PRINT
            printf("%d: %lu -- %lu\n", i, C[i], bufferC[i]);
#endif
        }
    }
    if (status) {
        printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }

    // Deallocation
    free(A);
    free(C);
    free(C2);
    vud_rank_free(&r);
	
    return status ? 0 : -1;
}
