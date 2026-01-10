/**
* app.c
* BS Host Application Source File
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

#if ENERGY
#include <dpu_probe.h>
#endif


#define NR_DPUS     64
#define NR_TASKLETS 16

#include <vud.h>
#include <vud_sk.h>
#include <vud_ime.h>
#include <vud_mem.h>
#include "../../src/vud_log.h"

#include "support/params.h"
#include "support/timer.h"
#include "support/common.h"
#include "support/prim_results.h"

#ifndef DPU_BINARY
#define DPU_BINARY "../bs"
#endif

#ifndef ARG_OFFSET
#define ARG_OFFSET   0x2000u
#endif
#define ARG_SIZE     ((uint32_t)sizeof(dpu_arguments_t))
#define A_OFFSET     (ARG_OFFSET + ((ARG_SIZE + 0xFFu) & ~0xFFu))

// Create input arrays
void create_test_file(DTYPE * input, DTYPE * querys, uint64_t  nr_elements, uint64_t nr_querys) {

	input[0] = 1;
	for (uint64_t i = 1; i < nr_elements; i++) {
		input[i] = input[i - 1] + 1;
	}
	for (uint64_t i = 0; i < nr_querys; i++) {
		querys[i] = i;
	}
}

// Compute output in the host
int64_t binarySearch(DTYPE * input, DTYPE * querys, DTYPE input_size, uint64_t num_querys)
{
	uint64_t result = -1;
	DTYPE r;
	for(uint64_t q = 0; q < num_querys; q++)
	{
		DTYPE l = 0;
		r = input_size;
		while (l <= r) {
			DTYPE m = l + (r - l) / 2;

			// Check if x is present at mid
			if (input[m] == querys[q])
			result = m;

			// If x greater, ignore left half
			if (input[m] < querys[q])
			l = m + 1;

			// If x is smaller, ignore right half
			else
			r = m - 1;
		}
	}
	//printf("CPU search: querys:%d, input_size%d, input:[%d, %d, %d]\n", num_querys, input_size, input[0], input[1], input[2]);
	return result;
}

static void push_args_array(vud_rank *r, dpu_arguments_t *args, uint32_t nr_of_dpus) {
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

	// Create the timer
	Timer timer;

	#if ENERGY
	struct dpu_probe_t probe;
	DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
	#endif

	uint64_t input_size = INPUT_SIZE;
        uint64_t num_querys = p.num_querys;

	// Query number adjustement for proper partitioning
        const uint64_t gran = (uint64_t)nr_of_dpus * NR_TASKLETS;
        if (num_querys % gran) num_querys += (gran - (num_querys % gran));
        assert((num_querys % gran) == 0 && "num_querys divisibility");
	printf("num_querys:%d\n", num_querys);

	DTYPE * input  = malloc((input_size) * sizeof(DTYPE));
	DTYPE * querys = malloc((num_querys) * sizeof(DTYPE));

	// Create an input file with arbitrary data
	create_test_file(input, querys, input_size, num_querys);

	// Compute host solution
	start(&timer, 0, 0);
	int64_t result_host = binarySearch(input, querys, input_size - 1, num_querys);
	stop(&timer, 0);

	// Create kernel arguments
	uint64_t slice_per_dpu          = num_querys / nr_of_dpus;
	//printf("slice_per_dpu:%d\n", slice_per_dpu);
	int64_t result_dpu;

	for (unsigned int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
		// Perform input transfers
		uint64_t i = 0;

		if (rep >= p.n_warmup)
		start(&timer, 1, rep - p.n_warmup);

                dpu_arguments_t args[NR_DPUS];
                for (uint32_t i = 0; i < nr_of_dpus; ++i) {
                    args[i].input_size    = input_size;
                    args[i].slice_per_dpu = slice_per_dpu;
                    args[i].kernel        = kernel1;
                }
                push_args_array(&r, args, nr_of_dpus);
                if (r.err) { fprintf(stderr, "args push failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
        
                /* 2) push input to A_OFFSET */
                const size_t input_bytes  = input_size * sizeof(DTYPE);
                const vud_mram_size words_input = (vud_mram_size)((input_bytes + 7) / 8);
                {
                    const uint64_t *lanes[NR_DPUS];
                    for (uint32_t i = 0; i < nr_of_dpus; ++i) lanes[i] = (const uint64_t*)input;
                    for (uint32_t i = nr_of_dpus; i < NR_DPUS;  ++i) lanes[i] = lanes[0];
                    vud_simple_transfer(&r, words_input, (const uint64_t (*)[NR_DPUS])&lanes, A_OFFSET);
                    if (r.err) { fprintf(stderr, "push input failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
                }
        
                /* 3) push queries right after input (per-DPU slice) */
                const size_t query_bytes_per_dpu = slice_per_dpu * sizeof(DTYPE);
                const vud_mram_size words_q = (vud_mram_size)((query_bytes_per_dpu + 7) / 8);
                const vud_mram_addr Q_base = (vud_mram_addr)(A_OFFSET + (vud_mram_addr)input_bytes);
                {
                    const uint64_t *lanes[NR_DPUS];
                    for (uint32_t i = 0; i < nr_of_dpus; ++i)
                        lanes[i] = (const uint64_t*)(querys + slice_per_dpu * i);
                    for (uint32_t i = nr_of_dpus; i < NR_DPUS; ++i) lanes[i] = lanes[0];
                    vud_simple_transfer(&r, words_q, (const uint64_t (*)[NR_DPUS])&lanes, Q_base);
                    if (r.err) { fprintf(stderr, "push queries failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
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

		if (rep >= p.n_warmup)
		{
			stop(&timer, 2);
			#if ENERGY
			DPU_ASSERT(dpu_probe_stop(&probe));
			#endif
		}
	vud_rank_rel_mux(&r);

	vud_ime_wait(&r);
		// Print logs if required
		#if PRINT
		unsigned int each_dpu = 0;
		printf("Display DPU Logs\n");
		DPU_FOREACH(dpu_set, dpu)
		{
			printf("DPU#%d:\n", each_dpu);
			DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
			each_dpu++;
		}
		#endif

		// Retrieve results
		if (rep >= p.n_warmup)
		start(&timer, 3, rep - p.n_warmup);
                uint64_t per_dpu_max64[NR_DPUS];
                {
                    uint64_t *ptrs[NR_DPUS];
                    for (uint32_t d = 0; d < nr_of_dpus; ++d) ptrs[d] = &per_dpu_max64[d];
                    for (uint32_t d = nr_of_dpus; d < NR_DPUS;  ++d) ptrs[d] = ptrs[0];
                    vud_simple_gather(&r, /*words=*/1, /*offset=*/SK_LOG_OFFSET, &ptrs);
                    if (r.err) { 
			    fprintf(stderr, "gather max failed: %s\n", vud_error_str(r.err)); 
			    return EXIT_FAILURE; 
		    }
                }
                result_dpu = -1;
                for (uint32_t d = 0; d < nr_of_dpus; ++d)
                    if ((int64_t)per_dpu_max64[d] > result_dpu) 
			    result_dpu = (int64_t)per_dpu_max64[d];
		if(rep >= p.n_warmup)
		stop(&timer, 3);
	}
	// Print timing results
	printf("CPU Version Time (ms): ");
	print(&timer, 0, p.n_reps);
	printf("CPU-DPU Time (ms): ");
	print(&timer, 1, p.n_reps);
	printf("DPU Kernel Time (ms): ");
	print(&timer, 2, p.n_reps);
	printf("DPU-CPU Time (ms): ");
	print(&timer, 3, p.n_reps);

        // update CSV
#define TEST_NAME "BS"
#define RESULTS_FILE "prim_results.csv"
        //update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 0, p.n_reps, "CPU");
        update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 1, p.n_reps, "M_C2D");
        update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 3, p.n_reps, "M_D2C");
        update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 2, p.n_reps, "DPU");

	#if ENERGY
	double energy;
	DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &energy));
	printf("DPU Energy (J): %f\t", energy * num_iterations);
	#endif

	int status = (result_dpu == result_host);
	if (status) {
		printf("\n[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] results are equal\n");
	} else {
		printf("\n[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] results differ!\n");
	}

	free(input);
	vud_rank_free(&r);

	return status ? 0 : 1;
}
