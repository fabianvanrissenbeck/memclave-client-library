/**
* app.c
* HST-S Host Application Source File
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
#include "support/prim_results.h"

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "../hsts"
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

#ifndef roundup
#define roundup(n, m)  (((n) % (m)) ? ((n) + ((m) - ((n) % (m)))) : (n))
#endif

static inline uint16_t ByteSwap16(uint16_t x) { return (uint16_t)((x << 8) | (x >> 8)); }


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

static void push_args_array(vud_rank* r, dpu_arguments_t* args, uint32_t nr_of_dpus) {
    const vud_mram_size words = (vud_mram_size)((sizeof(dpu_arguments_t) + 7u) / 8u);

    // Stage each lane into an aligned u64 buffer of `words` words.
    // Use static to avoid VLA on stack; adjust 1024 if your max DPUs differs.
    _Alignas(8) uint64_t staged[1024][4];  // 4 covers up to 32B struct; grow if needed
    assert(nr_of_dpus <= 1024);
    assert(words <= 4);

    for (uint32_t i = 0; i < nr_of_dpus; ++i) {
        memset(staged[i], 0, words * 8u);
        memcpy(staged[i], &args[i], sizeof(dpu_arguments_t));
    }

    // Build lane pointer array with u64* (what the API expects)
    const uint64_t* lanes[NR_DPUS];
    for (uint32_t i = 0; i < nr_of_dpus; ++i) lanes[i] = staged[i];
    for (uint32_t i = nr_of_dpus; i < NR_DPUS; ++i) lanes[i] = staged[0];  // pad

    vud_simple_transfer(r, words, (const uint64_t (*)[NR_DPUS])&lanes, ARG_OFFSET);
}


// Main of the Host Application
int main(int argc, char **argv) {

    struct Params p = input_params(argc, argv);

    uint32_t nr_of_dpus = NR_DPUS;
    
#if ENERGY
    struct dpu_probe_t probe;
    DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif

    vud_rank r = vud_rank_alloc(VUD_ALLOC_ANY);
    if (r.err) { 
            printf(stderr,"rank_alloc failed\n"); return EXIT_FAILURE; 
    }
    vud_ime_wait(&r);
    if (r.err) { 
            printf(stderr,"ime_wait failed\n"); return EXIT_FAILURE; 
    }
    printf("Allocated %d DPU(s)\n", nr_of_dpus);
    vud_ime_load(&r, "../hsts");
 
    if (r.err) {
        puts("cannot load subkernel");
        return -1;
    }
    vud_rank_nr_workers(&r, 12);
    if (r.err) { 
	    fprintf(stderr, "cannot start worker threads: %s\n", vud_error_str(r.err)); 
	    return EXIT_FAILURE; 
    }
 
    uint8_t key[32];
    //random_key(key);
 
    //vud_ime_install_key(&r, key, NULL, NULL);
 
    if (r.err) {
        puts("key exchange failed");
        return -1;
    }

    unsigned int i = 0;
    unsigned int input_size; // Size of input image
    unsigned int dpu_s = p.dpu_s;
    if(p.exp == 0)
        input_size = p.input_size * nr_of_dpus; // Size of input image
    else if(p.exp == 1)
        input_size = p.input_size; // Size of input image
    else
        input_size = p.input_size * dpu_s; // Size of input image

    //const unsigned int input_size_8bytes = ((input_size * sizeof(T)) % 8) != 0 ? roundup(input_size, 8) : input_size; // Input size per DPU (max.), 8-byte aligned
    const unsigned int input_size_dpu = divceil(input_size, nr_of_dpus); // Input size per DPU (max.)
    //const unsigned int input_size_dpu_8bytes = ((input_size_dpu * sizeof(T)) % 8) != 0 ? roundup(input_size_dpu, 8) : input_size_dpu; // Input size per DPU (max.), 8-byte aligned
    const unsigned int elem_per_8B = (8u / (unsigned)sizeof(T));
const unsigned int input_size_8bytes =
    (((size_t)input_size * sizeof(T)) % 8u) ? roundup(input_size, elem_per_8B) : input_size;
const unsigned int input_size_dpu_8bytes =
    (((size_t)input_size_dpu * sizeof(T)) % 8u) ? roundup(input_size_dpu, elem_per_8B) : input_size_dpu;


    // Input/output allocation
    A = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    //T *bufferA = A;
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

    // MRAM offsets (same across DPUs)
    const vud_mram_addr A_off      = (vud_mram_addr)A_OFFSET;
    const vud_mram_addr HISTO_off  = (vud_mram_addr)(A_off + (vud_mram_addr)(input_size_dpu_8bytes * sizeof(T)));

    // Per-DPU arguments (kept same field names as PRIM HST-S)
    dpu_arguments_t *input_arguments = (dpu_arguments_t*)malloc(nr_of_dpus * sizeof(*input_arguments));
    if (!input_arguments) { fprintf(stderr, "args alloc failed\n"); return EXIT_FAILURE; }

    // For all but the last DPU, `.size` is the exact element count for that DPU; the last gets the remainder.
    for (unsigned int i = 0; i < nr_of_dpus; i++) {
        unsigned int beg = i * input_size_dpu;
        unsigned int end = beg + input_size_dpu;
        if (end > input_size) end = input_size;
        unsigned int elems_i = (beg < input_size) ? (end - beg) : 0;

        // Store bytes for kernel convenience (fits the original PRIM semantics where size/transfer_size are bytes)
        input_arguments[i].size = elems_i * sizeof(T);
        // Not strictly needed for VUD uniform transfers, but kept for compatibility with your subkernel
        input_arguments[i].transfer_size = input_size_dpu_8bytes * sizeof(T);
        input_arguments[i].bins = p.bins;
        input_arguments[i].kernel = 0; // HST-S has one kernel
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
        // Input arguments
        push_args_array(&r, input_arguments, nr_of_dpus);

        // push A chunks per DPU at A_OFFSET (uniform length in words)
        {
            const vud_mram_size wordsA = (vud_mram_size)((input_size_dpu_8bytes * sizeof(T)) / 8u);
            const uint64_t* lanes[NR_DPUS];
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
        i = 0;
        if(rep >= p.n_warmup)
            start(&timer, 3, rep - p.n_warmup);

        {
            // Gather ceil(bins * 4 / 8) words per DPU, then copy into uint32 histo slots
            const size_t bytes_per_dpu = (size_t)p.bins * sizeof(unsigned int);
            const vud_mram_size wordsH = (vud_mram_size)((bytes_per_dpu + 7) / 8);
            // tmp buffer of u64 for all DPUs
            uint64_t* tmp = (uint64_t*)malloc((size_t)wordsH * 8 * nr_of_dpus);
            if (!tmp) { fprintf(stderr, "gather tmp alloc failed\n"); return EXIT_FAILURE; }

            uint64_t* lanes[NR_DPUS];
            for (unsigned int i = 0; i < nr_of_dpus; ++i)
                lanes[i] = &tmp[(size_t)i * wordsH];
            for (unsigned int i = nr_of_dpus; i < NR_DPUS; ++i)
                lanes[i] = lanes[0];

            vud_simple_gather(&r, wordsH, HISTO_off, (uint64_t* (*)[NR_DPUS])&lanes);
            if (r.err) { free(tmp); fprintf(stderr, "gather failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

            // Copy each DPU's bytes into the uint32 histo array
            for (unsigned int i = 0; i < nr_of_dpus; ++i) {
                memcpy(&histo[(size_t)i * p.bins], &tmp[(size_t)i * wordsH], bytes_per_dpu);
            }
            free(tmp);
        }
        if(rep >= p.n_warmup)
            stop(&timer, 3);

    }
    // After vud_simple_gather and the memcpy() that fills histo[i * bins ...]
    {
    // Optional sanity: per-DPU sum check before merge
    const uint64_t expected = (uint64_t)input_arguments[0].size / sizeof(T);
    for (uint32_t i = 0; i < nr_of_dpus; ++i) {
        uint64_t sum = 0;
        for (uint32_t j = 0; j < p.bins; ++j)
            sum += histo[(size_t)i * p.bins + j];
        if (sum != expected) {
            fprintf(stderr, "!! Host sees DPU[%u] sum=%llu, expected=%llu\n",
                    i, sum, expected);
        }
    }

    // Merge all DPUs into histo[0..bins-1]
    for (uint32_t i = 1; i < nr_of_dpus; ++i) {
        unsigned int *src = &histo[(size_t)i * p.bins];
        for (uint32_t j = 0; j < p.bins; ++j)
            histo[j] += src[j];
    }
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
#define TEST_NAME "HST-S"
#define RESULTS_FILE "prim_results.csv"
    //update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 0, p.n_reps, "CPU");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 1, p.n_reps, "M_C2D");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 3, p.n_reps, "M_D2C");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 2, p.n_reps, "DPU");

    #if ENERGY
    double energy;
    DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &energy));
    printf("DPU Energy (J): %f\t", energy);
    #endif	

    // Check output
    bool status = true;
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
		int i = 0;
            if(nr_of_dpus * histo_host[j] != histo[j]){ 
                status = false;
//#if PRINT
                printf("%u - %u: %u -- %u\n", j, j, nr_of_dpus * histo_host[j], histo[j]);
//#endif
		i++;
		if (i >10) break;
            }
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
        count_sum  = logs[d * WORDS_PER_DPU + 4];
        histo_dpu[0]  = logs[d * WORDS_PER_DPU + 1];
        histo_dpu[1]  = logs[d * WORDS_PER_DPU + 2];
        histo_dpu[2]  = logs[d * WORDS_PER_DPU + 3];
        printf("DPU[%d] magic:%llu bins:%llu size:%llu transfer_size:%llu count_sum:%llu histo_dpu: [%llu, %llu, %llu]\n", d, magic, bins, size, transfer_size, count_sum, histo_dpu[0], histo_dpu[1], histo_dpu[2]);
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
