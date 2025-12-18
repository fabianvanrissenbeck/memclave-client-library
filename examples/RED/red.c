/**
* app.c
* RED Host Application Source File
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
#define DPU_BINARY "../red"
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
static T* A;

// Create input arrays
static void read_input(T* A, unsigned int nr_elements) {
    srand(0);
    printf("nr_elements\t%u\t", nr_elements);
    for (unsigned int i = 0; i < nr_elements; i++) {
        A[i] = (T)(rand());
    }
}

// Compute output in the host
static T reduction_host(T* A, unsigned int nr_elements) {
    T count = 0;
    for (unsigned int i = 0; i < nr_elements; i++) {
        count += A[i];
    }
    return count;
}

static void push_args_array(vud_rank *r, dpu_arguments_t *args, uint32_t nr_of_dpus) {
    const vud_mram_size words = (vud_mram_size)((sizeof(dpu_arguments_t) + 7u) / 8u);
    _Alignas(8) uint64_t staged[NR_DPUS][4]; /* up to 32B dpu_arguments_t */
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

    const uint32_t nr_of_dpus = NR_DPUS;    

#if ENERGY
    struct dpu_probe_t probe;
    DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif

    printf("Allocated %d DPU(s)\n", nr_of_dpus);

    unsigned int i = 0;
#if PERF
    double cc = 0;
    double cc_min = 0;
#endif

    const unsigned int input_size = p.exp == 0 ? p.input_size * nr_of_dpus : p.input_size; // Total input size (weak or strong scaling)
    const unsigned int input_size_8bytes = 
        ((input_size * sizeof(T)) % 8) != 0 ? roundup(input_size, 8) : input_size; // Input size per DPU (max.), 8-byte aligned
    const unsigned int input_size_dpu = divceil(input_size, nr_of_dpus); // Input size per DPU (max.)
    const unsigned int input_size_dpu_8bytes = 
        ((input_size_dpu * sizeof(T)) % 8) != 0 ? roundup(input_size_dpu, 8) : input_size_dpu; // Input size per DPU (max.), 8-byte aligned
    const size_t bytes_per_dpu = (size_t)input_size_dpu_8bytes * sizeof(T);


    // Input/output allocation
    A = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    T *bufferA = A;
    T count = 0;
    T count_host = 0;

    // Create an input file with arbitrary data
    read_input(A, input_size);

    // Timer declaration
    Timer timer;

    printf("NR_TASKLETS\t%d\tBL\t%d\n", NR_TASKLETS, BL);

    bool ok = true;
    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Compute output on CPU (performance comparison and verification purposes)
        if(rep >= p.n_warmup)
            start(&timer, 0, rep - p.n_warmup);
        count_host = reduction_host(A, input_size);
        if(rep >= p.n_warmup)
            stop(&timer, 0);

        printf("Load input data\n");
        if(rep >= p.n_warmup)
            start(&timer, 1, rep - p.n_warmup);
        count = 0;
        // Input arguments
        dpu_arguments_t args0[NR_DPUS];
        for (uint32_t i = 0; i < nr_of_dpus; ++i) {
            args0[i].size    = (uint32_t)bytes_per_dpu;
            args0[i].kernel  = 0;    /* single-kernel reduction */
            args0[i].t_count = (T)0; /* unused here */
        }
	{
 	   const unsigned int tail_elems = input_size_8bytes - input_size_dpu_8bytes * (NR_DPUS - 1);  // elems on last DPU
    	   const size_t tail_bytes = (size_t)tail_elems * sizeof(T);
    	   args0[NR_DPUS - 1].size = (uint32_t)tail_bytes;
	}
        push_args_array(&r, args0, nr_of_dpus);
        if (r.err) { fprintf(stderr, "args transfer failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

        {
            const vud_mram_size wordsA = (vud_mram_size)((bytes_per_dpu + 7) / 8);
            const uint64_t *lanes[NR_DPUS];
            for (uint32_t i = 0; i < nr_of_dpus; ++i)
                lanes[i] = (const uint64_t*)(A + (size_t)i * input_size_dpu_8bytes);
            for (uint32_t i = nr_of_dpus; i < NR_DPUS; ++i) lanes[i] = lanes[0];
            vud_simple_transfer(&r, wordsA, (const uint64_t (*)[NR_DPUS])&lanes, A_OFFSET);
            if (r.err) { fprintf(stderr, "push A failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
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

        printf("Retrieve results\n");
        uint64_t per_dpu_totals[NR_DPUS];
        {
            uint64_t *ptrs[NR_DPUS];
            for (uint32_t d = 0; d < nr_of_dpus; ++d) 
		    ptrs[d] = &per_dpu_totals[d];
            for (uint32_t d = nr_of_dpus; d < NR_DPUS; ++d) 
		    ptrs[d] = ptrs[0];
            vud_simple_gather(&r, /*words=*/1, /*offset=*/SK_LOG_OFFSET, &ptrs);
            if (r.err) { 
		    fprintf(stderr, "gather totals failed: %s\n", vud_error_str(r.err)); 
		    return EXIT_FAILURE; 
	    }
        }
        for (uint32_t d = 0; d < nr_of_dpus; ++d) 
		count += (T)per_dpu_totals[d];

#if PERF
        DPU_FOREACH(dpu_set, dpu, i) {
            results[i].cycles = 0;
            // Retrieve tasklet timings
            for (unsigned int each_tasklet = 0; each_tasklet < NR_TASKLETS; each_tasklet++) {
                if (results_retrieve[i][each_tasklet].cycles > results[i].cycles)
                    results[i].cycles = results_retrieve[i][each_tasklet].cycles;
            }
            free(results_retrieve[i]);
        }
#endif
        if(rep >= p.n_warmup)
            stop(&timer, 3);
	if (count != count_host) 
		ok = false;

#if PERF
        uint64_t max_cycles = 0;
        uint64_t min_cycles = 0xFFFFFFFFFFFFFFFF;
        // Print performance results
        if(rep >= p.n_warmup){
            i = 0;
            DPU_FOREACH(dpu_set, dpu) {
                if(results[i].cycles > max_cycles)
                    max_cycles = results[i].cycles;
                if(results[i].cycles < min_cycles)
                    min_cycles = results[i].cycles;
                i++;
            }
            cc += (double)max_cycles;
            cc_min += (double)min_cycles;
        }
#endif

    }
#if PERF
    printf("DPU cycles  = %g cc\n", cc / p.n_reps);
#endif

    // Print timing results
    printf("CPU ");
    print(&timer, 0, p.n_reps);
    printf("CPU-DPU ");
    print(&timer, 1, p.n_reps);
    printf("DPU Kernel ");
    print(&timer, 2, p.n_reps);
    printf("Inter-DPU ");
    print(&timer, 3, p.n_reps);

    // update CSV
#define TEST_NAME "RED"
#define RESULTS_FILE "prim_results.csv"
    //update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 0, p.n_reps, "CPU");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 1, p.n_reps, "M_C2D");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 2, p.n_reps, "DPU");

    #if ENERGY
    double energy;
    DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &energy));
    printf("DPU Energy (J): %f\t", energy);
    #endif	

    // Check output
    bool status = true;
    if(count != count_host) status = false;
    if (status) {
        printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }

    // Deallocation
    free(A);
    vud_rank_free(&r);
	
    return status ? 0 : -1;
}
