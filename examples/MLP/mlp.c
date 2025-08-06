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

#if ENERGY
#include <dpu_probe.h>
#endif

#define NR_DPUS 64

#include "../../src/vud.h"
#include "../../src/vud_mem.h"
#include "../../src/vud_ime.h"
#include "../../src/vud_log.h"
#include "support/common.h"    // defines T, dpu_arguments_t, dpu_results_t, etc.
#include "support/params.h"    // parses command-line into struct Params
#include "support/timer.h"     // for timing host vs DPU if you want

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "../mlp.sk"
#endif

#define ARG_OFFSET     0x2000
#define ARG_SIZE       sizeof(dpu_arguments_t)
#define A_OFFSET       (ARG_OFFSET + ((ARG_SIZE + 0xFF) & ~0xFF))

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

// Main of the Host Application
int main(int argc, char **argv) {

	struct Params p = input_params(argc, argv);

	uint32_t nr_of_dpus = NR_DPUS;

	unsigned int i, l;
	unsigned int m_size = p.m_size;
	unsigned int n_size = p.n_size;

        vud_rank r = vud_rank_alloc(0);
        if (r.err) { 
                printf(stderr,"rank_alloc failed\n"); return EXIT_FAILURE; 
        }
	vud_ime_wait(&r);
        if (r.err) { 
                printf(stderr,"ime_wait failed\n"); return EXIT_FAILURE; 
        }

	// Initialize help data
	dpu_info = (struct dpu_info_t *) malloc(nr_of_dpus * sizeof(struct dpu_info_t));
	dpu_arguments_t *input_args = (dpu_arguments_t *) malloc(nr_of_dpus * sizeof(dpu_arguments_t));
	uint32_t max_rows_per_dpu = 0;
	uint32_t n_size_pad = n_size;
	if(n_size % 2 == 1){
		n_size_pad++;
	}

	// Timer
	Timer timer;
	i = 0;
	for(uint32_t i = 0; i < nr_of_dpus; i++) {
		uint32_t rows_per_dpu;
		uint32_t prev_rows_dpu = 0;
		uint32_t chunks = m_size / nr_of_dpus;
		rows_per_dpu = chunks;
		uint32_t rest_rows = m_size % nr_of_dpus;
		if (i < rest_rows)
			rows_per_dpu++;
		if (rest_rows > 0) {
			if (i >= rest_rows)
				prev_rows_dpu = rest_rows * (chunks + 1) + (i - rest_rows) * chunks;
			else
				prev_rows_dpu = i * (chunks + 1);
		} else {
			prev_rows_dpu = i * chunks;
		}

		// Keep max rows for parallel transfers
		uint32_t rows_per_dpu_pad = rows_per_dpu;
		if (rows_per_dpu_pad % 2 == 1) // 4-byte elements
			rows_per_dpu_pad++;
		if (rows_per_dpu_pad > max_rows_per_dpu)
			max_rows_per_dpu = rows_per_dpu_pad;

		dpu_info[i].rows_per_dpu = rows_per_dpu;
		dpu_info[i].rows_per_dpu_pad = rows_per_dpu_pad;
		dpu_info[i].prev_rows_dpu = prev_rows_dpu;

		// Copy input arguments to DPU
		input_args[i].n_size = n_size;
		input_args[i].n_size_pad = n_size_pad;
		input_args[i].nr_rows = rows_per_dpu;
		input_args[i].max_rows = max_rows_per_dpu;
	}

	A = (T**)malloc(NUM_LAYERS * sizeof(T*));
	for(l = 0; l < NUM_LAYERS; l++)
		A[l] = (T*)malloc( max_rows_per_dpu * nr_of_dpus * n_size_pad * sizeof(T));


	B = (T*)malloc(n_size * sizeof(T));
	B_host = (T*)malloc(n_size * sizeof(T));
	C = (T*)malloc(m_size * sizeof(T));
	C_dpu = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
	B_tmp = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));

	init_data(A, B, B_host, m_size, n_size);
	printf("after init_data\n");

	// Compute output on CPU (performance comparison and verification purposes)
	start(&timer, 0, 0);
	mlp_host(C, A, B_host, m_size, n_size);
	stop(&timer, 0);
	printf("after mlp_host\n");

        // precompute byte & word sizes
        size_t   slice_bytes = max_rows_per_dpu * n_size_pad * sizeof(T);
        size_t    vec_bytes  =           n_size_pad * sizeof(T);
        size_t  arg_words    = (sizeof(dpu_arguments_t) + 7) / 8;
        size_t slice_words   = slice_bytes      / 8;
        size_t  vec_words    = vec_bytes        / 8;
	// MRAM offsets
        vud_mram_addr A_off = A_OFFSET;
        vud_mram_addr B_off = A_off + slice_bytes;
        vud_mram_addr C_off = B_off + vec_bytes;
	printf("before for loop\n");
#if 1
	for (unsigned int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
		if (rep >= p.n_warmup)
			start(&timer, 1, rep - p.n_warmup);
#if 0
		// Input arguments
		i = 0;
		// Copy input arguments to DPU
		DPU_FOREACH(dpu_set, dpu, i) {
			input_args[i].max_rows = max_rows_per_dpu;
			DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
		}
		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));
#endif
	       printf("before push dpu_arguments\n");
	        // 1) push the dpu_arguments_t array
                {
                    // build per‐DPU pointers:
                    const dpu_arguments_t* ptrs[NR_DPUS];
                    for (unsigned i = 0; i < nr_of_dpus; i++)
                        ptrs[i] = &input_args[i];
                    // dummy‐fill any extra slots
                    for (unsigned i = nr_of_dpus; i < NR_DPUS; i++)
                        ptrs[i] = ptrs[0];
  
                    vud_simple_transfer(&r,
                                        arg_words,
                                        (const uint64_t (*)[NR_DPUS])&ptrs,
                                        ARG_OFFSET);
                    if (r.err) { 
                            printf(stderr,"ime_wait failed\n"); return EXIT_FAILURE; 
                    }
                }


#if 0
		// Copy input array and vector
		i = 0;
		DPU_FOREACH(dpu_set, dpu, i) {
			DPU_ASSERT(dpu_prepare_xfer(dpu, A[0] + dpu_info[i].prev_rows_dpu * n_size));
		}
		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, max_rows_per_dpu * n_size_pad * sizeof(T), DPU_XFER_DEFAULT));
		i = 0;
		DPU_FOREACH(dpu_set, dpu, i) {
			DPU_ASSERT(dpu_prepare_xfer(dpu, B));
		}
		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * n_size_pad * sizeof(T) , n_size_pad * sizeof(T), DPU_XFER_DEFAULT));
#endif
	       printf("before scatter first layers weight matrix\n");
		// 2) scatter the first layer’s weight‐matrix slice
                {
                    uint64_t* ptrs[NR_DPUS];
                    for (unsigned i = 0; i < nr_of_dpus; i++) {
                        // A[0] is the base of layer‐0; prev_rows_dpu gives row offset
                        T* base = (uint64_t*)(A[0] + dpu_info[i].prev_rows_dpu * n_size);
                        ptrs[i]  = (uint64_t*) base;
                    }
                    for (unsigned i = nr_of_dpus; i < NR_DPUS; i++)
                        ptrs[i] = ptrs[0];
  
                    vud_simple_transfer(&r,
                                        slice_words,
					(const uint64_t (*)[NR_DPUS])&ptrs,
                                        A_off);
                }
		
	       printf("before broadcast input vec B\n");
                // 3) broadcast the input vector B
                {
                    uint64_t* ptrs[NR_DPUS];
                    for (unsigned i = 0; i < nr_of_dpus; i++)
                        ptrs[i] = (uint64_t*) B;
                    for (unsigned i = nr_of_dpus; i < NR_DPUS; i++)
                        ptrs[i] = ptrs[0];
         
                    vud_simple_transfer(&r,
                                        vec_words,
					(const uint64_t (*)[NR_DPUS])&ptrs,
                                        B_off);
                }
		if (rep >= p.n_warmup)
			stop(&timer, 1);

		// Run kernel on DPUs
		if (rep >= p.n_warmup)
		{
			start(&timer, 2, rep - p.n_warmup);
#if ENERGY
			DPU_ASSERT(dpu_probe_start(&probe));
#endif
		}

	       printf("before launch sk\n");
		//DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
		vud_ime_launch_sk(&r, DPU_BINARY);
                if (r.err) { 
                        printf(stderr,"launch failed\n"); return EXIT_FAILURE; 
                }
                vud_ime_wait(&r);
                if (r.err) { 
                        printf(stderr,"wait failed\n"); return EXIT_FAILURE; 
                }

		if (rep >= p.n_warmup)
		{
			stop(&timer, 2);
#if ENERGY
			DPU_ASSERT(dpu_probe_stop(&probe));
#endif
		}

	       printf("before lay for loop sk\n");
		for(int lay = 1; lay < NUM_LAYERS; lay++){
			if (rep >= p.n_warmup)
				start(&timer, 4, rep - p.n_warmup);
			i = 0;

#if 0
			// Copy C_dpu
			DPU_FOREACH(dpu_set, dpu, i) {
				DPU_ASSERT(dpu_prepare_xfer(dpu, C_dpu + i * max_rows_per_dpu));
			}
			DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * n_size_pad * sizeof(T) + n_size_pad * sizeof(T), max_rows_per_dpu * sizeof(T), DPU_XFER_DEFAULT));
#endif
	       		printf("before copy C_dpu\n");
			/* Gather output back from DPUs into C_dpu */
                        {
                            uint64_t *ptrs[NR_DPUS];
                            /* point each slot to the right place in C_dpu */
                            for (unsigned i = 0; i < nr_of_dpus; ++i) {
                                ptrs[i] = (uint64_t*)(C_dpu + i * max_rows_per_dpu);
                            }
                            /* dummy-fill remaining slots so we don’t walk off the end */
                            for (unsigned i = nr_of_dpus; i < NR_DPUS; ++i) {
                                ptrs[i] = ptrs[0];
                            }
                            vud_simple_gather(&r,
                                              max_rows_per_dpu,  /* words per-DPU block */
                                              C_off,
                                              (uint64_t(*)[NR_DPUS])&ptrs);
                        }
	       		printf("after copy C_dpu\n");

			// B = C
			unsigned int n, j;
			i = 0;
			for (n = 0; n < nr_of_dpus; n++) {
				for (j = 0; j < dpu_info[n].rows_per_dpu; j++) {
					B_tmp[i] = C_dpu[n * max_rows_per_dpu + j];
					i++;
				}
			}
#if 0
			i = 0;
			DPU_FOREACH(dpu_set, dpu, i) {
				DPU_ASSERT(dpu_prepare_xfer(dpu, B_tmp));
			}
			DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * n_size_pad * sizeof(T) , n_size_pad * sizeof(T), DPU_XFER_DEFAULT));

			// Copy next matrix of weights
			i = 0;
			DPU_FOREACH(dpu_set, dpu, i) {
				DPU_ASSERT(dpu_prepare_xfer(dpu, A[lay] + dpu_info[i].prev_rows_dpu * n_size));
			}
			DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, max_rows_per_dpu * n_size_pad * sizeof(T), DPU_XFER_DEFAULT));
#endif
		printf("before copy B_tmp\n");
	    {
                uint64_t* ptrs[NR_DPUS];
                for (unsigned i = 0; i < nr_of_dpus; i++)
                    ptrs[i] = (uint64_t*) B_tmp;
                for (unsigned i = nr_of_dpus; i < NR_DPUS; i++)
                    ptrs[i] = ptrs[0];

                vud_simple_transfer(&r,
                                    vec_words,
                                    (const uint64_t (*)[NR_DPUS])&ptrs,
                                    B_off);
            }

		printf("before scater next layer weight matrix\n");
            // 6) scatter next layer’s weight‐matrix slice A[lay]
            {
                uint64_t* ptrs[NR_DPUS];
                for (unsigned i = 0; i < nr_of_dpus; i++) {
                    T* base = A[lay] + dpu_info[i].prev_rows_dpu * n_size;
                    ptrs[i]  = (uint64_t*) base;
                }
                for (unsigned i = nr_of_dpus; i < NR_DPUS; i++)
                    ptrs[i] = ptrs[0];

                vud_simple_transfer(&r,
                                    slice_words,
                                    (const uint64_t (*)[NR_DPUS])&ptrs,
                                    A_off);
            }

			if(rep >= p.n_warmup)
				stop(&timer, 4);

			if (rep >= p.n_warmup)
			{
				start(&timer, 2, rep - p.n_warmup);
#if ENERGY
				DPU_ASSERT(dpu_probe_start(&probe));
#endif
			}

			printf("before launch sk again\n");
			//DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
		        vud_ime_launch_sk(&r, DPU_BINARY);
                        if (r.err) { 
                                printf(stderr,"launch failed\n"); return EXIT_FAILURE; 
                        }
                        vud_ime_wait       (&r);
                        if (r.err) { 
                                printf(stderr,"wait failed\n"); return EXIT_FAILURE; 
                        }

			if (rep >= p.n_warmup)
			{
				stop(&timer, 2);
#if ENERGY
				DPU_ASSERT(dpu_probe_stop(&probe));
#endif
			}
		}

#if PRINT
		// Display DPU Logs
		DPU_FOREACH(dpu_set, dpu) {
			DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
		}
#endif

#if 0
		// Retrieve results
		if (rep >= p.n_warmup)
			start(&timer, 3, rep - p.n_warmup);
		i = 0;
		DPU_FOREACH(dpu_set, dpu, i) {
			DPU_ASSERT(dpu_prepare_xfer(dpu, C_dpu + i * max_rows_per_dpu));
		}
		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * n_size_pad * sizeof(T) + n_size_pad * sizeof(T), max_rows_per_dpu * sizeof(T), DPU_XFER_DEFAULT));
#endif
			printf("before gather\n");
		{
                uint64_t *ptrs[NR_DPUS];
                // point each slot to the right place in C_dpu
                for (unsigned i = 0; i < nr_of_dpus; i++) {
                    ptrs[i] = (uint64_t*)(C_dpu + i * max_rows_per_dpu);
                }
                // dummy‐fill any extra slots
                for (unsigned i = nr_of_dpus; i < NR_DPUS; i++) {
                    ptrs[i] = ptrs[0];
                }
                vud_simple_gather(&r,
                                  max_rows_per_dpu,     /* words per‐DPU block */
                                  C_off,
                                  (uint64_t(*)[NR_DPUS])&ptrs);
                }
		if(rep >= p.n_warmup)
			stop(&timer, 3);
	}
#endif

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

#if ENERGY
	printf("Energy (J): %f J\t", avg_energy);
#endif
	printf("\n\n");

	// Check output
	bool status = true;
	unsigned int n, j;
	i = 0;
	for (n = 0; n < nr_of_dpus; n++) {
		for (j = 0; j < dpu_info[n].rows_per_dpu; j++) {
			if(C[i] != C_dpu[n * max_rows_per_dpu + j]) {
				status = false;
#if PRINT
				printf("%d: %d -- %d\n", i, C[i], C_dpu[n * max_rows_per_dpu + j]);
#endif
			}
			i++;
		}
	}
	if (status) {
		printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
	} else {
		printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
	}

	// Deallocation
	for(i = 0; i < NUM_LAYERS; i++)
		free(A[i]);
	free(A);
	free(B);
	free(C);
	free(C_dpu);
        vud_rank_free(&r);

#if ENERGY
	DPU_ASSERT(dpu_probe_deinit(&probe));
#endif

	return status ? 0 : -1;
}
