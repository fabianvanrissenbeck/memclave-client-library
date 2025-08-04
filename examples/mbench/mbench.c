#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>

#include "../../src/vud.h"
#include "../../src/vud_mem.h"
#include "../../src/vud_ime.h"
#include "../../src/vud_log.h"
#include "support/common.h"    // defines T, dpu_arguments_t, dpu_results_t, etc.
#include "support/params.h"    // parses command-line into struct Params
#include "support/timer.h"     // for timing host vs DPU if you want

#define VEC_SIZE 64
// Must match the subkernel’s defines:
#define MRAM_SIZE_BYTES   (64u << 20)
#define DEBUG_OFFSET      (MRAM_SIZE_BYTES - 64)
#define NUM_DEBUG_WORDS   3

#define NR_TASKLETS 1
//#define ADD 1
#define SUB 1
#define NR_DPUS 64
#define TEST_DPUS 8
// Pointer declaration
static T* A;
static T* B;
static T* Z;
static T* C2;

// Create input arrays
static void read_input(T* A, T* B, T* Z, unsigned int nr_elements) {
    srand(0);
    printf("nr_elements\t%u\t", nr_elements);
    for (unsigned int i = 0; i < nr_elements; i++) {
        A[i] = (T) (rand()+1);
        B[i] = (T) (rand());
        Z[i] = (T) (0);
    }
}

// Compute output in the host
static void update_host(T* C, T* A, unsigned int nr_elements) {
    //printf("nr_elements:%d\n", nr_elements);
    for (unsigned int i = 0; i < nr_elements; i++) {
#if ADD
        C[i] = A[i] + (nr_elements / NR_DPUS);
#elif SUB
        C[i] = A[i] - (nr_elements / TEST_DPUS);
#elif MUL
        C[i] = A[i] * (nr_elements / NR_DPUS);
#elif DIV
        C[i] = A[i] / (nr_elements / NR_DPUS);
#endif
    }
    printf("expected: A[0]=%x C[0]=%x\n", A[0], C[0]);
    printf("expected: A[%d]=%x C[1]=%x\n", nr_elements/TEST_DPUS, A[nr_elements/TEST_DPUS], C[nr_elements/TEST_DPUS]);
    printf("expected: A[%d]=%x C[2]=%x\n", 2*nr_elements/TEST_DPUS, A[2*nr_elements/TEST_DPUS], C[2*nr_elements/TEST_DPUS]);
}

#define ARG_OFFSET   0x1000       // safe, at MRAM base
#define DATA_OFFSET  0x1100      // leave 256 B for your args
#define OUTPUT_OFFSET  (0x12000)      // leave 256 B for your args
#define RESULT_OFFSET  (0x23000)      // leave 256 B for your args   

int main(int argc, char** argv) {
    uint32_t nr_of_dpus = TEST_DPUS;
    struct Params p = input_params(argc, argv);
    if (optind + 2 > argc) {
        //printf("Usage: dpurun <core loader> <mram image> [options...]\n");
        //return EXIT_FAILURE;
    }

    vud_rank r = vud_rank_alloc(0);

    if (r.err) {
        puts("Cannot allocate rank.");
        return 1;
    }

    vud_ime_wait(&r);

    if (r.err) {
        puts("cannot wait for rank");
        goto error;
    }

    unsigned int i = 0;
    double cc = 0;
    double cc_min = 0;
    const unsigned int input_size = p.exp == 0 ? p.input_size * nr_of_dpus : p.input_size;

    // Input/output allocation
    A = malloc(input_size * sizeof(T));
    B = malloc(input_size * sizeof(T));
    Z = malloc(input_size * sizeof(T));
    T *bufferA = A;
    T *bufferB = B;
    T *bufferZ = Z;
    C2 = malloc(input_size * sizeof(T));

    // Create an input file with arbitrary data
    read_input(A, B, Z, input_size);

    // Timer declaration
    Timer timer;

    printf("\tBL\t%d\n", BL);
    uint64_t *zptrs[NR_DPUS];
    const unsigned int input_size_dpu = input_size / nr_of_dpus;
    for (int d = 0; d < nr_of_dpus; ++d) {
        // cast bufferA to uint64_t* so we step in 8‑byte increments
        zptrs[d] = (uint64_t*)((uint8_t *)bufferZ 
                      + (size_t)d * input_size_dpu * sizeof(T));
    }
    for (int d = nr_of_dpus; d < NR_DPUS; ++d) {
        // cast bufferA to uint64_t* so we step in 8‑byte increments
        zptrs[d] = (uint64_t*)((uint8_t *)bufferZ );
    }

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
        size_t size = (input_size_dpu * sizeof(uint64_t) + 7) / 8;
        printf("inside for size:%d, input_size_dpu:%d, input_size:%d\n", size, input_size_dpu, input_size);
        vud_simple_transfer(&r,
                            size,
                            &zptrs,
                            DATA_OFFSET);
        vud_simple_transfer(&r,
                            size,
                            &zptrs,
                            OUTPUT_OFFSET);
	printf("Zero complete\n");

        // Compute output on CPU (performance comparison and verification purposes)
        if(rep >= p.n_warmup)
            start(&timer, 0, rep - p.n_warmup);
        update_host(C2, A, input_size);
        if(rep >= p.n_warmup)
            stop(&timer, 0);
        //printf("%d: C2[0] %x\n", 0, C2[0]);

        printf("Load input data\n");
        if(rep >= p.n_warmup)
            start(&timer, 1, rep - p.n_warmup);
        // Input arguments
        unsigned int kernel = 0;
        dpu_arguments_t input_arguments = {input_size_dpu * sizeof(T), kernel};
	printf("args: %x\n", input_arguments.size);
        size_t arg_words = (sizeof(input_arguments) + 7) / 8;
        vud_broadcast_transfer(&r, arg_words, (const void *)&input_arguments, ARG_OFFSET);
        // Copy input arrays
        i = 0;
        uint64_t *host_ptrs[NR_DPUS];
        for (int d = 0; d < nr_of_dpus; ++d) {
            // cast bufferA to uint64_t* so we step in 8‑byte increments
            host_ptrs[d] = (uint64_t*)((uint8_t *)bufferA 
                          + (size_t)d * input_size_dpu * sizeof(T));
        }
    for (int d = nr_of_dpus; d < NR_DPUS; ++d) {
        // cast bufferA to uint64_t* so we step in 8‑byte increments
        host_ptrs[d] = (uint64_t*)((uint8_t *)bufferZ );
    }
        // 2) compute # of 8‑byte words per DPU
        size_t data_words = (input_size_dpu * sizeof(uint64_t) + 7) / 8;
        printf("before simple_transfer\n");
        vud_simple_transfer(&r,
                            data_words,
                            &host_ptrs,   // note the &
                            DATA_OFFSET);
        printf("After simple_transfer\n");
        if(rep >= p.n_warmup)
            stop(&timer, 1);

        printf("Run program on DPU(s) %d\n", nr_of_dpus);
        // Run DPU kernel
        if(rep >= p.n_warmup)
            start(&timer, 2, rep - p.n_warmup);
        vud_ime_launch_sk(&r, "../add.sk");
	vud_ime_wait(&r);
        if(rep >= p.n_warmup)
            stop(&timer, 2);
#if PRINT
        {
            unsigned int each_dpu = 0;
            printf("Display DPU Logs\n");
            DPU_FOREACH (set, dpu) {
                printf("DPU#%d:\n", each_dpu);
                DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
                each_dpu++;
            }
        }
#endif

        printf("Retrieve results\n");
        if(rep >= p.n_warmup)
            start(&timer, 3, rep - p.n_warmup);
        dpu_results_t results[nr_of_dpus];
        i = 0;
        uint64_t *dsts[NR_DPUS];
        for (int d = 0; d < nr_of_dpus; ++d) {
            dsts[d] = (uint64_t *)((uint8_t *)bufferB + (size_t)d * input_size_dpu * sizeof(T));
        }
    for (int d = nr_of_dpus; d < NR_DPUS; ++d) {
        // cast bufferA to uint64_t* so we step in 8‑byte increments
        dsts[d] = (uint64_t*)((uint8_t *)bufferZ );
    }
	size_t words = (input_size_dpu * sizeof(T) + 7) / 8;
	vud_simple_gather(&r, words, OUTPUT_OFFSET, &dsts);
        printf("Retrieve results complete\n");
			
#if PERF
            results[i].cycles = 0;
            // Retrieve tasklet timings
            for (unsigned int each_tasklet = 0; each_tasklet < NR_TASKLETS; each_tasklet++) {
                dpu_results_t result;
                result.cycles = 0;
                DPU_ASSERT(dpu_copy_from_mram(dpu.dpu, &result, RESULT_OFFSET, sizeof(dpu_results_t)));
                if (result.cycles > results[i].cycles)
                    results[i].cycles = result.cycles;
            }
#endif
            i++;
        //}
        //printf("%d: C2[0] %x -- buffer %x (cycles:%x)\n", i, C2[0], bufferB[0], results[0].cycles);
        if(rep >= p.n_warmup)
            stop(&timer, 3);

#if PERF
        uint64_t max_cycles = 0;
        uint64_t min_cycles = 0xFFFFFFFFFFFFFFFF;
        // Print performance results
        if(rep >= p.n_warmup){
            i = 0;
            DPU_FOREACH(set, dpu) {
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
#ifdef ADD
    printf("ADD\n");
#elif SUB
    printf("SUB\n");
#elif MUL
    printf("MUL\n");
#elif DIV
    printf("DIV\n");
#endif
    printf("DPU cycles  = %g cc\n", cc / p.n_reps);

    // Print timing results
    printf("CPU ");
    print(&timer, 0, p.n_reps);
    printf("CPU-DPU ");
    print(&timer, 1, p.n_reps);
    printf("DPU Kernel ");
    print(&timer, 2, p.n_reps);
    printf("DPU-CPU ");
    print(&timer, 3, p.n_reps);

    // Check output
    bool status = true;
    for (i = 0; i < input_size; i++) {
        if(C2[i] != bufferB[i]){ 
            status = false;
//#if PRINT
            printf("%d: %x -- %x\n", i, C2[i], bufferB[i]);
//#endif
	break;
        }
    }
    if (status) {
        printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }

    // Deallocation
    free(A);
    free(B);
    free(C2);

    //uint64_t logs[64][SK_LOG_MAX_ENTRIES];
    //int err = vud_log_read(&r, 64, logs);
    uint64_t logs[NR_DPUS];
    uint64_t *ptrs[NR_DPUS];
    for (int d = 0; d < nr_of_dpus; ++d) {
        ptrs[d] = &logs[d];
    }
    for (int d = nr_of_dpus; d < NR_DPUS; ++d) {
        // cast bufferA to uint64_t* so we step in 8‑byte increments
        ptrs[d] = (uint64_t*)((uint8_t *)bufferZ );
    }
    int err = 0;
    size_t nw = 1;                          // number of 8‑byte words
    vud_simple_gather(&r,
                  nw,
                  DATA_OFFSET,        // byte offset in MRAM
                  &ptrs);             // note the ‘&’ here
    if (err) {
        printf("Log gather failed: %d\n", r.err);
    } else {
        for (int d = 0; d < NR_DPUS; ++d) {
            uint64_t v = (uint64_t)logs[d];
            //printf("DPU %02d first element = %x\n", d, v);
        }
    }
    // OUTPUT
    vud_simple_gather(&r,
                  nw,
                  OUTPUT_OFFSET,        // byte offset in MRAM
                  &ptrs);             // note the ‘&’ here
    printf("Output \n");
    if (err) {
        printf("Log gather failed: %d\n", r.err);
    } else {
        for (int d = 0; d < NR_DPUS; ++d) {
            uint64_t v = (uint64_t)logs[d];
            printf("DPU %02d first element = %x\n", d, v);
        }
    }
    vud_rank_free(&r);
    return 0;

error:
    printf("VUD Error %d\n", r.err);
    vud_rank_free(&r);
    return 1;
}
