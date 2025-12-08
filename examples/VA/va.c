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

#define NR_DPUS 64

#include <vud.h>
#include <vud_sk.h>
#include <vud_ime.h>
#include <vud_mem.h>
#include "../../src/vud_log.h"
#include "support/common.h"
#include "support/timer.h"
#include "support/params.h"

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "../va"
#endif

#ifndef ARG_OFFSET
#define ARG_OFFSET 0x1000
#endif
#ifndef DATA_OFFSET
#define DATA_OFFSET 0x2000
#endif
#ifndef OUTPUT_OFFSET
#define OUTPUT_OFFSET (0x200000)
#endif
#ifndef RESULT_OFFSET
#define RESULT_OFFSET (0x400000)
#endif
#ifndef SK_LOG_OFFSET
#define SK_LOG_OFFSET ((64u << 20) - 64u)
#endif

#if ENERGY
#include <dpu_probe.h>
#endif

// Pointer declaration
static T* A;
static T* B;
static T* C;
static T* C2;

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
static inline void push_args_array(vud_rank* r, dpu_arguments_t* args, uint32_t nr_of_dpus) {
    const size_t words = (sizeof(dpu_arguments_t) + 7u) / 8u;
    const uint64_t* ptrs[NR_DPUS];
    for (uint32_t i = 0; i < nr_of_dpus; ++i) ptrs[i] = (const uint64_t*)&args[i];
    for (uint32_t i = nr_of_dpus; i < NR_DPUS; ++i) ptrs[i] = ptrs[0];
    vud_simple_transfer(r, words, (const uint64_t* (*)[NR_DPUS])&ptrs, ARG_OFFSET);
}

// Main of the Host Application
int main(int argc, char **argv) {

    struct Params p = input_params(argc, argv);

    vud_rank r = vud_rank_alloc(VUD_ALLOC_ANY);
    if (r.err) {
	    fprintf(stderr, "vud_rank_alloc failed (%d)\n", r.err); return EXIT_FAILURE; 
    }
    vud_ime_wait(&r); // ensure MUX is exposed
    if (r.err) { 
	    fprintf(stderr, "vud_ime_wait failed (%d)\n", r.err); return EXIT_FAILURE; 
    }
    uint32_t nr_of_dpus = NR_DPUS;
    unsigned int i = 0;

#if ENERGY
    struct dpu_probe_t probe;
    DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif

    // Allocate DPUs and load binary

    const unsigned int input_size = p.input_size; // VA typically uses 1-D length
    const unsigned int input_size_8bytes = (((unsigned long long)input_size * sizeof(T)) % 8ULL) ? roundup(input_size, 8) : input_size;
    const unsigned int input_size_dpu = divceil(input_size, nr_of_dpus);
    const unsigned int input_size_dpu_8bytes = (((unsigned long long)input_size_dpu * sizeof(T)) % 8ULL) ? roundup(input_size_dpu, 8) : input_size_dpu;

    // Input/output allocation
    A = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    B = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    C = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    C2 = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    T *bufferA = A;
    T *bufferB = B;
    T *bufferC = C2;

    // Create an input file with arbitrary data
    read_input(A, B, input_size);

    vud_ime_load(&r, DPU_BINARY);
    if (r.err) { fprintf(stderr, "vud_ime_load_sk failed (%d)\n", r.err); return EXIT_FAILURE; }
 
    uint8_t key[32];
    //random_key(key);
 
    //vud_ime_install_key(&r, key, NULL, NULL);
 
    if (r.err) {
        puts("key exchange failed");
        return -1;
    }

    // Timer declaration
    Timer timer;

    printf("NR_DPU\t%d\tBL\t%d\n", NR_DPUS, BL);

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Compute output on CPU (performance comparison and verification purposes)
        if(rep >= p.n_warmup)
            start(&timer, 0, rep - p.n_warmup);


        // Prepare input arguments per DPU — keep names and shape
        unsigned int kernel = 0;
        dpu_arguments_t input_arguments[NR_DPUS];
        for (i = 0; i < nr_of_dpus - 1; i++) {
                input_arguments[i].size = input_size_dpu_8bytes * sizeof(T);
                input_arguments[i].transfer_size = input_size_dpu_8bytes * sizeof(T);
                input_arguments[i].kernel = kernel;
        }
        input_arguments[nr_of_dpus - 1].size = (input_size_8bytes - input_size_dpu_8bytes * (NR_DPUS - 1)) * sizeof(T);
        input_arguments[nr_of_dpus - 1].transfer_size = input_size_dpu_8bytes * sizeof(T);
        input_arguments[nr_of_dpus - 1].kernel = kernel;


        // 1) push arguments
        if (rep >= p.n_warmup) start(&timer, 1, rep - p.n_warmup);
	for (i = 0; i < nr_of_dpus; i++) {
            const uint32_t sz = input_arguments[i].size; // bytes
            // A…B gap
            assert(DATA_OFFSET + sz <= OUTPUT_OFFSET && "A/B overlap: increase OUTPUT_OFFSET");
            // B…C gap
            assert(OUTPUT_OFFSET + sz <= RESULT_OFFSET && "B/C overlap: increase RESULT_OFFSET");
            // C…end-of-MRAM (optional safety)
            assert(RESULT_OFFSET + sz <= (64u<<20) && "C runs past MRAM");
        }

	//printf("\n before push_args_array");
        push_args_array(&r, input_arguments, nr_of_dpus);
	//printf("\n after push_args_array");
#if 0
{
    const size_t arg_bytes = ((sizeof(dpu_arguments_t) + 7u) & ~7u);
    const size_t arg_words = arg_bytes / 8u;
    uint64_t *args_back[64];
    uint64_t buf[2] = {0};
    for (unsigned j = 0; j < 64; ++j) args_back[j] = buf;
    vud_simple_gather(&r, arg_words, ARG_OFFSET, (uint64_t* (*)[64])&args_back);
    dpu_arguments_t arg0; memcpy(&arg0, buf, sizeof(arg0));
    fprintf(stderr, "Args[DPU0]: size=%u xfer=%u kernel=%u\n",
            arg0.size, arg0.transfer_size, arg0.kernel);
}
#endif

        // 2) scatter input vectors A and B to MRAM
	size_t words = (input_size_dpu_8bytes * sizeof(T)) / 8;
        const uint64_t* pA[64];
        const uint64_t* pB[64];
        for (unsigned j = 0; j < nr_of_dpus; ++j) {
                pA[j] = (const uint64_t*)(bufferA + input_size_dpu_8bytes * j);
                pB[j] = (const uint64_t*)(bufferB + input_size_dpu_8bytes * j);
        }
	for (unsigned j = nr_of_dpus; j < 64; ++j) { pA[j] = pA[0]; pB[j] = pB[0]; }

	//printf("\n before simple_transfer DATA_OFFSET");
        vud_simple_transfer(&r, words, (const uint64_t* (*)[NR_DPUS])&pA, DATA_OFFSET);
	//printf("\n after simple_transfer DATA_OFFSET");
        vud_simple_transfer(&r, words, (const uint64_t* (*)[NR_DPUS])&pB, OUTPUT_OFFSET);
	//printf("\n after simple_transfer OUTPUT_OFFSET");
#if 0
{
    size_t words = (input_size_dpu_8bytes * sizeof(T)) / 8;

    // one real buffers for lane 0
    T *A0 = malloc(input_size_dpu_8bytes * sizeof(T));
    T *B0 = malloc(input_size_dpu_8bytes * sizeof(T));

    // scratch sink for all other lanes so they don't overwrite A0/B0
    T *Ascratch = malloc(input_size_dpu_8bytes * sizeof(T));
    T *Bscratch = malloc(input_size_dpu_8bytes * sizeof(T));

    uint64_t *Aback[64], *Bback[64];
    for (unsigned j = 0; j < 64; ++j) {
        Aback[j] = (j == 0) ? (uint64_t*)A0 : (uint64_t*)Ascratch;
        Bback[j] = (j == 0) ? (uint64_t*)B0 : (uint64_t*)Bscratch;
    }

    vud_simple_gather(&r, words, DATA_OFFSET,   (uint64_t* (*)[64])&Aback);
    vud_simple_gather(&r, words, OUTPUT_OFFSET, (uint64_t* (*)[64])&Bback);

    // spot check some indices inside the lane-0 slice
    for (unsigned k = 0; k < 4; ++k) {
        unsigned idx = k * 1024;                   // any indices < input_size_dpu_8bytes
        if (A[idx] != A0[idx] || B[idx] != B0[idx]) {
            fprintf(stderr, "Lane0 scatter mismatch @%u: A %u!=%u  B %u!=%u\n",
                idx, (unsigned)A[idx], (unsigned)A0[idx],
                     (unsigned)B[idx], (unsigned)B0[idx]);
            break;
        }
    }

    free(A0); free(B0); free(Ascratch); free(Bscratch);
}
#endif

        //}
        if (rep >= p.n_warmup) stop(&timer, 1);
        // 3) launch + wait
        if (rep >= p.n_warmup) start(&timer, 2, rep - p.n_warmup);
        vud_ime_launch(&r);
        vud_ime_wait(&r);
        if (rep >= p.n_warmup) stop(&timer, 2);

#if 0
{
    uint64_t *logbufs[64];
    uint64_t d0log[8], scratch[8];
    for (unsigned j = 0; j < 64; ++j) logbufs[j] = (j == 0) ? d0log : scratch;

    vud_simple_gather(&r, 8 /* qwords */, SK_LOG_OFFSET, (uint64_t* (*)[64])&logbufs);

    fprintf(stderr,
        "DPU0-ARGS: magic=%016llx size=%u xfer=%u kernel=%u done=%llu\n",
        (unsigned long long)d0log[0],
        (unsigned)d0log[1],
        (unsigned)d0log[2],
        (unsigned)d0log[3],
        (unsigned long long)d0log[7]);
}
{
    // read SK log
    uint64_t *logbufs[64];
    uint64_t d0log[8], scratch[8];
    for (unsigned j = 0; j < 64; ++j) logbufs[j] = (j == 0) ? d0log : scratch;
    vud_simple_gather(&r, 8, SK_LOG_OFFSET, (uint64_t* (*)[64])&logbufs);

    // read the first 16 B of C back
    uint64_t *Cback[64];
    uint64_t C0_16B[2], Cscratch[2];
    for (unsigned j = 0; j < 64; ++j) Cback[j] = (j == 0) ? C0_16B : Cscratch;
    vud_simple_gather(&r, 2 /*words*/, RESULT_OFFSET, (uint64_t* (*)[64])&Cback);

    // expected sums from our host buffers
    uint64_t exp0 = ((uint64_t*)A)[0] + ((uint64_t*)B)[0];
    uint64_t exp1 = ((uint64_t*)A)[1] + ((uint64_t*)B)[1];

    fprintf(stderr,
        "DPU0-PROBE: magic=%016llx a0=%llu a1=%llu b0=%llu b1=%llu c0=%llu c1=%llu | exp0=%llu exp1=%llu\n",
        (unsigned long long)d0log[0],
        (unsigned long long)d0log[1], (unsigned long long)d0log[2],
        (unsigned long long)d0log[3], (unsigned long long)d0log[4],
        (unsigned long long)d0log[5], (unsigned long long)d0log[6],
        (unsigned long long)exp0,      (unsigned long long)exp1);

    fprintf(stderr, "C0[0..1]=%llu %llu\n",
        (unsigned long long)C0_16B[0], (unsigned long long)C0_16B[1]);
}
#endif



        // 4) gather result vector C from MRAM
        if (rep >= p.n_warmup) start(&timer, 3, rep - p.n_warmup);
        {
                size_t words = (input_size_dpu_8bytes * sizeof(T)) / 8;
                uint64_t* pC[NR_DPUS];
                for (unsigned j = 0; j < nr_of_dpus; ++j)
                        pC[j] = (uint64_t*)(bufferC + input_size_dpu_8bytes * j);
                for (unsigned j = nr_of_dpus; j < NR_DPUS; ++j)
                        pC[j] = pC[0];
                vud_simple_gather(&r, words, RESULT_OFFSET, (uint64_t* (*)[NR_DPUS])&pC);
        }
        if (rep >= p.n_warmup) stop(&timer, 3);


        if (rep >= p.n_warmup) stop(&timer, 0);
        }

        // Verification on host
        vector_addition_host(C, A, B, input_size);
        bool status = true;
        for (unsigned i = 0; i < input_size; i++) {
                if (C[i] != C2[i]) { 
			status = false; 
			printf("c[%d]:%d != C2[%d]:%d\n",i, C[i], i, C2[i]);
			break; 
		}
        }
        if (status)
                printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
        else
                printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");


        // Report performance (same buckets as original timer usage)
        printf("CPU (ms): ");
        //print(&timer, p.n_reps);
{
    int nb = (int)nr_of_dpus;              // NOT NR_DPUS if that’s just the cap
    #define LANES 64
    #define WORDS_PER_DPU 8  // we read 1 x 8B per DPU
	    uint64_t logs[LANES * WORDS_PER_DPU];
    uint64_t* ptrs[LANES];
    for (int d = 0; d < nr_of_dpus; ++d) {
        ptrs[d] = &logs[d*WORDS_PER_DPU];
    }
    vud_simple_gather(&r, WORDS_PER_DPU, SK_LOG_OFFSET, &ptrs);
    uint64_t max_cycles = 0;
    uint64_t magic;
    uint64_t total;
    uint64_t done ;
    uint64_t tasks = 0;   
    uint64_t T_BL = 0;
    for (int d = 0; d < nb; ++d) {
        magic = logs[d * WORDS_PER_DPU + 0];
        total = logs[d * WORDS_PER_DPU + 1];
        done  = logs[d * WORDS_PER_DPU + 7];
        tasks  = logs[d * WORDS_PER_DPU + 5];
        T_BL  = logs[d * WORDS_PER_DPU + 6];
        if (magic == 0xffffLL && done == 1ULL) {
            if (total > max_cycles) max_cycles = total;
        }
        //printf("DPU[%d] logs: %llu %llu %llu\n", d, magic, total, done);
    }
    printf("DPU cycles (whole-kernel, max over DPUs): %llu %llu %llu %llu\n",
           (unsigned long long) max_cycles, magic, tasks, T_BL);


}


        // Deallocation
        free(A); free(B); free(C); free(C2);
        vud_rank_free(&r);
        return status ? 0 : -1;
}
