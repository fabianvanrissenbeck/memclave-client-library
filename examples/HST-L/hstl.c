/**
* app.c
* HST-L Host Application Source File
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>

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
#define DPU_BINARY "../hstl"
#endif

#if ENERGY
#include <dpu_probe.h>
#endif

#ifndef ARG_OFFSET
#define ARG_OFFSET   0x2000u
#endif
#define ARG_SIZE     ((uint32_t)sizeof(dpu_arguments_t))
#define A_OFFSET     (ARG_OFFSET + ((ARG_SIZE + 0xFFu) & ~0xFFu))

// Pointer declaration
static T* A;
static unsigned int* histo_host;
static unsigned int* histo;

// Create input arrays
static void read_input(T* A, const Params p) {

    char  dctFileName[100];
    FILE *File = NULL;

    // Open input file
    unsigned short temp;
    sprintf(dctFileName, p.file_name);
    if((File = fopen(dctFileName, "rb")) != NULL) {
        for(unsigned int y = 0; y < p.input_size; y++) {
            fread(&temp, sizeof(unsigned short), 1, File);
            A[y] = (unsigned int)ByteSwap16(temp);
            if(A[y] >= 4096)
                A[y] = 4095;
        }
        fclose(File);
    } else {
        printf("%s does not exist\n", dctFileName);
        exit(1);
    }
}

// Compute output in the host
static void histogram_host(unsigned int* histo, T* A, unsigned int bins, unsigned int nr_elements, int exp, unsigned int nr_of_dpus) {
    if(!exp){
        for (unsigned int i = 0; i < nr_of_dpus; i++) {
            for (unsigned int j = 0; j < nr_elements; j++) {
                T d = A[j];
                histo[i * bins + ((d * bins) >> DEPTH)] += 1;
            }
        }
    }
    else{
        for (unsigned int j = 0; j < nr_elements; j++) {
            T d = A[j];
            histo[(d * bins) >> DEPTH] += 1;
        }
    }
}

static void push_args_array(vud_rank *r, dpu_arguments_t *args, uint32_t nr_of_dpus) {
    const vud_mram_size words = (vud_mram_size)((sizeof(dpu_arguments_t) + 7u) / 8u);
    _Alignas(8) uint64_t staged[1024][4];   // supports up to 32B structs; grow if you extend args
    assert(nr_of_dpus <= 1024);
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
    uint32_t nr_of_dpus = NR_DPUS;

    vud_rank r = vud_rank_alloc(VUD_ALLOC_ANY);
    if (r.err) { fprintf(stderr, "rank_alloc failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
    vud_ime_wait(&r);
    if (r.err) { fprintf(stderr, "ime_wait failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

    vud_ime_load(&r, DPU_BINARY);
    if (r.err) { fprintf(stderr, "cannot load subkernel: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

    uint8_t key[32];
    //random_key(key);
 
    //vud_ime_install_key(&r, key, NULL, NULL);
 
    if (r.err) {
        puts("key exchange failed");
        return -1;
    }
    
#if ENERGY
    struct dpu_probe_t probe;
    DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif

    printf("Allocated %d DPU(s)\n", nr_of_dpus);

    unsigned int i = 0;
    unsigned int input_size; // Size of input image
    unsigned int dpu_s = p.dpu_s;
    if(p.exp == 0)
        input_size = p.input_size * nr_of_dpus; // Size of input image
    else if(p.exp == 1)
        input_size = p.input_size; // Size of input image
	else
        input_size = p.input_size * dpu_s; // Size of input image

    const unsigned int input_size_8bytes = 
        ((input_size * sizeof(T)) % 8) != 0 ? roundup(input_size, 8) : input_size; // Input size per DPU (max.), 8-byte aligned
    const unsigned int input_size_dpu = divceil(input_size, nr_of_dpus); // Input size per DPU (max.)
    const unsigned int input_size_dpu_8bytes = 
        ((input_size_dpu * sizeof(T)) % 8) != 0 ? roundup(input_size_dpu, 8) : input_size_dpu; // Input size per DPU (max.), 8-byte aligned

    // Input/output allocation
    A = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    T *bufferA = A;
    histo_host = malloc(p.bins * sizeof(unsigned int));
    histo = malloc(nr_of_dpus * p.bins * sizeof(unsigned int));

    // Create an input file with arbitrary data
    read_input(A, p);
    if(p.exp == 0){
        for(unsigned int j = 1; j < nr_of_dpus; j++){
            memcpy(&A[j * input_size_dpu_8bytes], &A[0], input_size_dpu_8bytes * sizeof(T));
        }
    }
    else if(p.exp == 2){
        for(unsigned int j = 1; j < dpu_s; j++)
            memcpy(&A[j * p.input_size], &A[0], p.input_size * sizeof(T));
    }

    const vud_mram_addr A_off     = (vud_mram_addr)A_OFFSET;
    const vud_mram_addr HISTO_off = (vud_mram_addr)(A_off + (vud_mram_addr)(input_size_dpu_8bytes * sizeof(T)));

    dpu_arguments_t *input_arguments = (dpu_arguments_t*)malloc(nr_of_dpus * sizeof(*input_arguments));
    if (!input_arguments) { fprintf(stderr, "args alloc failed\n"); return EXIT_FAILURE; }

    for (unsigned int i = 0; i < nr_of_dpus; i++) {
        unsigned int beg = i * input_size_dpu;
        unsigned int end = beg + input_size_dpu;
        if (end > input_size) end = input_size;
        unsigned int elems_i = (beg < input_size) ? (end - beg) : 0;
        input_arguments[i].size          = elems_i * sizeof(T);
        input_arguments[i].transfer_size = input_size_dpu_8bytes * sizeof(T);
        input_arguments[i].bins          = p.bins;
        input_arguments[i].kernel        = 0;
    }

    // Timer declaration
    Timer timer;

    printf("NR_TASKLETS\t%d\tBL\t%d\tinput_size\t%u\n", NR_TASKLETS, BL, input_size);

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
        memset(histo_host, 0, p.bins * sizeof(unsigned int));
        memset(histo, 0, nr_of_dpus * p.bins * sizeof(unsigned int));

        // Compute output on CPU (performance comparison and verification purposes)
        if(rep >= p.n_warmup)
            start(&timer, 0, rep - p.n_warmup);
        histogram_host(histo_host, A, p.bins, p.input_size, 1, nr_of_dpus);
        if(rep >= p.n_warmup)
            stop(&timer, 0);

        printf("Load input data\n");
        if(rep >= p.n_warmup)
            start(&timer, 1, rep - p.n_warmup);
        push_args_array(&r, input_arguments, nr_of_dpus);

        {
            const vud_mram_size wordsA = (vud_mram_size)((input_size_dpu_8bytes * sizeof(T)) / 8u);
            const uint64_t *lanes[NR_DPUS];
            for (unsigned int i = 0; i < nr_of_dpus; ++i)
                lanes[i] = (const uint64_t*)(&A[(size_t)i * input_size_dpu_8bytes]);
            for (unsigned int i = nr_of_dpus; i < NR_DPUS; ++i)
                lanes[i] = lanes[0];
            vud_simple_transfer(&r, wordsA, (const uint64_t (*)[NR_DPUS])&lanes, A_off);
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
        if (r.err) { fprintf(stderr, "launch failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
        vud_ime_wait(&r);
	if (r.err) { fprintf(stderr, "wait failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

        if(rep >= p.n_warmup) {
            stop(&timer, 2);
            #if ENERGY
            DPU_ASSERT(dpu_probe_stop(&probe));
            #endif
        }

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
        i = 0;
        if(rep >= p.n_warmup)
            start(&timer, 3, rep - p.n_warmup);
	{
            const size_t bytes_per_dpu = (size_t)p.bins * sizeof(unsigned int);
            const vud_mram_size wordsH = (vud_mram_size)((bytes_per_dpu + 7) / 8);
            uint64_t *tmp = (uint64_t*)malloc((size_t)wordsH * 8 * nr_of_dpus);
            if (!tmp) { fprintf(stderr, "gather tmp alloc failed\n"); return EXIT_FAILURE; }

            uint64_t *lanes[NR_DPUS];
            for (unsigned int i = 0; i < nr_of_dpus; ++i)
                lanes[i] = &tmp[(size_t)i * wordsH];
            for (unsigned int i = nr_of_dpus; i < NR_DPUS; ++i)
                lanes[i] = lanes[0];

            vud_simple_gather(&r, wordsH, HISTO_off, (uint64_t* (*)[NR_DPUS])&lanes);
            if (r.err) { free(tmp); fprintf(stderr, "gather failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

            for (unsigned int i = 0; i < nr_of_dpus; ++i)
                memcpy(&histo[(size_t)i * p.bins], &tmp[(size_t)i * wordsH], bytes_per_dpu);
            free(tmp);
        }	
	for (uint32_t i = 0; i < nr_of_dpus; ++i) {
            uint64_t sum = 0;
            for (uint32_t j = 0; j < p.bins; ++j)
                sum += histo[(size_t)i * p.bins + j];
            uint64_t expected = (uint64_t)input_arguments[i].size / sizeof(T);
            if (sum != expected) {
                fprintf(stderr, "DPU[%u] sum=%llu expected=%llu  (ratio=%.2f)\n",
                        i, (unsigned long long)sum, (unsigned long long)expected,
                        expected ? (double)sum/(double)expected : 0.0);
            }
        }
        // Final histogram merging
	for (uint32_t i = 1; i < nr_of_dpus; ++i) {
            unsigned int *src = &histo[(size_t)i * p.bins];
            for (uint32_t j = 0; j < p.bins; ++j) histo[j] += src[j];
        }
        if(rep >= p.n_warmup)
            stop(&timer, 3);

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

    #if ENERGY
    double energy;
    DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &energy));
    printf("DPU Energy (J): %f\t", energy);
    #endif	


    // Check output
    bool status = true;
    volatile int itr = 0;
    if(p.exp == 1) 
        for (unsigned int j = 0; j < p.bins; j++) {
            if(histo_host[j] != histo[j]){ 
                status = false;
#if PRINT
                printf("%u - %u: %u -- %u\n", j, j, histo_host[j], histo[j]);
#endif
            }
        }
    else if(p.exp == 2) 
        for (unsigned int j = 0; j < p.bins; j++) {
            if(dpu_s * histo_host[j] != histo[j]){ 
                status = false;
#if PRINT
                printf("%u - %u: %u -- %u\n", j, j, dpu_s * histo_host[j], histo[j]);
#endif
            }
        }
    else
        for (unsigned int j = 0; j < p.bins; j++) {
            if(nr_of_dpus * histo_host[j] != histo[j]){ 
                status = false;
//#if PRINT
                printf("%u - %u: %u -- %u\n", j, j, nr_of_dpus * histo_host[j], histo[j]);
		itr++;
//#endif
            }
		if (itr >10) break;
        }
    if (status) {
        printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }

#if 0
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
    uint64_t bins;
    uint64_t size;
    uint64_t transfer_size = 0;
    uint64_t count_sum = 0;
    uint64_t histo_dpu[3];
    for (int d = 0; d < nb; ++d) {
        magic = logs[d * WORDS_PER_DPU + 0];
        bins = logs[d * WORDS_PER_DPU + 1];
        size  = logs[d * WORDS_PER_DPU + 2];
        transfer_size  = logs[d * WORDS_PER_DPU + 3];
        //count_sum  = logs[d * WORDS_PER_DPU + 4];
        //histo_dpu[0]  = logs[d * WORDS_PER_DPU + 1];
        //histo_dpu[1]  = logs[d * WORDS_PER_DPU + 2];
        //histo_dpu[2]  = logs[d * WORDS_PER_DPU + 3];
        printf("DPU[%d] magic:%llu bins:%llu size:%llu transfer_size:%llu \n", d, magic, bins, size, transfer_size);
    }
    }
#endif

    // Deallocation
    free(A);
    free(histo_host);
    free(histo);
    vud_rank_free(&r);
	
    return status ? 0 : -1;
}
