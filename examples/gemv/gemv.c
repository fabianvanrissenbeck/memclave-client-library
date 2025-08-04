#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>    // for getopt() & optind
#include <getopt.h>    // for getopt_long(), if you ever need it
#include <stdbool.h>

#include "../../src/vud.h"
#include "../../src/vud_mem.h"
#include "../../src/vud_ime.h"
#include "../../src/vud_log.h"
#include "support/common.h"    // defines T, dpu_arguments_t, dpu_results_t, etc.
#include "support/params.h"    // parses command-line into struct Params
#include "support/timer.h"     // for timing host vs DPU if you want

#define ARG_OFFSET     0x2000         // leave MRAM[0x0000–0x0FFF] for anything else
#define ARG_SIZE       sizeof(dpu_arguments_t)  // e.g. 16

// Next free address, aligned up to 0x100 boundary
#define A_OFFSET       (ARG_OFFSET + ((ARG_SIZE + 0xFF) & ~0xFF))  // e.g. 0x1100
#define NR_DPUS 64

static T* A;
static T* B;
static T* C;
static T* C_dpu;

// Create input arrays
static void init_data(T* A, T* B, unsigned int m_size, unsigned int n_size) {
	srand(0);

	for (unsigned int i = 0; i < m_size * n_size; i++)
	{
		A[i] = (unsigned int) (rand()%50);
	}

	for (unsigned int i = 0; i < n_size; i++)
	{
		B[i] = (unsigned int) (rand()%50);
	}
}

// Compute output in the host
static void gemv_host(T* C, T* A, T* B, unsigned int m_size, unsigned int n_size) {
	for (unsigned int i = 0; i < m_size; i++)
	{
		C[i] = 0;
	}

	for (unsigned int m = 0; m < m_size; m++) {
		for (unsigned int n = 0; n < n_size; n++)
		{
			C[m] += A[m * n_size + n] * B[n];
		}
	}
	printf("HOST  C[0] = %llu\n", (unsigned long long)C[0]);
}

void* load_file_complete(const char* path, size_t* out_size) {
    FILE* fp = fopen(path, "rb");
    printf("%s\n", path);
    assert(fp != NULL);

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    void* res = malloc(size);
    assert(res != NULL);

    fread(res, size, 1, fp);
    fclose(fp);

    *out_size = size;
    return res;
}

void buf_to_stdout(size_t sz, const uint64_t* buf) {
    FILE* p = popen("xxd -e -g 8", "w");
    assert(p != NULL);

    fwrite(buf, 1, sz * sizeof(buf[0]), p);
    pclose(p);
}

int main(int argc, char** argv) {
    uint32_t nr_of_dpus;

    struct Params p = input_params(argc, argv);
    if (optind + 2 > argc) {
        printf("Usage: dpurun <core loader> <mram image> [options...]\n");
        return EXIT_FAILURE;
    }
    printf("\n %s - %s - %s - %s - %s\n", argv[0], argv[1], argv[2], argv[3], argv[4]);
    printf("HOST sizeof(T) = %zu-bytes\n", sizeof(T));
    vud_rank r = vud_rank_alloc(0);
    if (r.err) { 
	    printf(stderr,"rank_alloc failed\n"); return EXIT_FAILURE; 
    }
    vud_ime_wait(&r);
    nr_of_dpus = 16;//NR_DPUS;
#if ENERGY
	struct dpu_probe_t probe;
	DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif

	unsigned int i;
	unsigned int m_size = p.m_size;
	unsigned int n_size = p.n_size;

	// Initialize help data
	dpu_info = (struct dpu_info_t *) malloc(nr_of_dpus * sizeof(struct dpu_info_t));
	dpu_arguments_t *input_args = (dpu_arguments_t *) malloc(nr_of_dpus * sizeof(dpu_arguments_t));
	uint32_t max_rows_per_dpu = 0;
	uint32_t n_size_pad = n_size;
	if(n_size % 2 == 1)
	{
		n_size_pad++;
	}

	i = 0;
        printf("before input_args compute, nr_of_dpus:%d\n", nr_of_dpus);
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
        printf("HOST args: n_size=%llu, n_size_pad=%llu, nr_rows=%llu, max_rows=%llu\n",
           n_size, n_size_pad, input_args[0].nr_rows, max_rows_per_dpu);
        	printf("dpu_info[i].prev_rows_dpu:%x\n", dpu_info[0].prev_rows_dpu);
        	printf("dpu_info[i].rows_per_dpu:%x\n", dpu_info[0].rows_per_dpu);
        	printf("dpu_info[i].rows_per_dpu_pad:%x\n", dpu_info[0].rows_per_dpu_pad);
        	printf("input_args[i].n_size:%x\n", input_args[0].n_size);
        printf("After input_args compute\n");

	A = malloc(max_rows_per_dpu * nr_of_dpus * n_size_pad * sizeof(T));
	B = malloc(n_size_pad * sizeof(T));
	C = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
	size_t slice_bytes = max_rows_per_dpu * n_size_pad * sizeof(T);
	size_t vec_bytes = n_size_pad * sizeof(T);
	size_t A_offset = A_OFFSET;
	size_t B_offset  = A_offset + slice_bytes;          // immediately after A
	size_t C_offset   = B_offset + vec_bytes; // right after B

	// Initialize data with arbitrary data
	init_data(A, B, m_size, n_size);

	// Timer
	Timer timer;

        printf("Before host compute\n");
	// Compute output on CPU (performance comparison and verification purposes)
	start(&timer, 0, 0);
	gemv_host(C, A, B, m_size, n_size);
	stop(&timer, 0);
	printf("ARG_OFFSET:%x, A_offset:%x, B_offset:%x, C_offset:%x", ARG_OFFSET, A_offset, B_offset, C_offset);
        printf("After host compute\n");
	for (unsigned int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
		if (rep >= p.n_warmup)
			start(&timer, 1, rep - p.n_warmup);
            // 1) Broadcast the per‐DPU argument structs
            {
                size_t arg_words = (sizeof(dpu_arguments_t) + 7) / 8;
                // we only need the first nr_of_dpus entries
                const dpu_arguments_t* ptrs[NR_DPUS];
                for (unsigned i = 0; i < nr_of_dpus; ++i) {
                    ptrs[i] = &input_args[i];
                }
                // dummy-fill the rest so we don’t walk off the end
                for (unsigned i = nr_of_dpus; i < NR_DPUS; ++i) {
                    ptrs[i] = &input_args[0];
                }
                vud_simple_transfer(&r,
                                    arg_words,
                                    (const void*)&ptrs,
                                    ARG_OFFSET);
            }
		//DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));
//	    printf("A[0]=%llu, A[1]=%llu, A[2]=%llu, A[3]=%llu\n",
//               A[0], A[1], A[2], A[3]);
        // 2) Scatter the matrix‐slice for each DPU
        {
            // one 8‐byte word per entry
            size_t slice_words = slice_bytes / 8;
            uint64_t* ptrs[NR_DPUS];
            for (unsigned i = 0; i < nr_of_dpus; ++i) {
                ptrs[i] = (uint64_t*)(A
                    + dpu_info[i].prev_rows_dpu * n_size);
            }
            // dummy-fill
            for (unsigned i = nr_of_dpus; i < NR_DPUS; ++i) {
                ptrs[i] = ptrs[0];
            }
            vud_simple_transfer(&r,
                                slice_words,
                                &ptrs,
                                A_offset);
        }
//printf("B:%llu\n",*(B));
//printf("B[1]:%llu\n",B[1]);
//printf("B[2]:%llu\n",B[2]);
//printf("B[3]:%llu\n",B[3]);

        // 3) Broadcast the vector B to every DPU
        {
            size_t vec_words = vec_bytes / 8;
            uint64_t* ptrs[NR_DPUS];
            for (unsigned i = 0; i < nr_of_dpus; ++i) {
                ptrs[i] = (uint64_t*)B;
            }
            // dummy-fill
            for (unsigned i = nr_of_dpus; i < NR_DPUS; ++i) {
                ptrs[i] = ptrs[0];
            }
            vud_simple_transfer(&r,
                                vec_words,
                                &ptrs,
                                B_offset);
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

                vud_ime_launch_sk(&r, "../gemv.sk");
    		if (r.err) { 
	    		printf("Launch failed %d\n", r.err);
			return EXIT_FAILURE; 
    		}
	        vud_ime_wait(&r);

		if (rep >= p.n_warmup)
		{
			stop(&timer, 2);
#if ENERGY
			DPU_ASSERT(dpu_probe_stop(&probe));
#endif
		}
#if PRINT
		// Display DPU Logs
		DPU_FOREACH(dpu_set, dpu) {
			DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
		}
#endif


		// Retrieve results
		C_dpu = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
		if (rep >= p.n_warmup)
			start(&timer, 3, rep - p.n_warmup);
            {
                // number of 8-byte words per DPU‐block
                size_t result_words = (max_rows_per_dpu * sizeof(T) + 7) / 8;
                uint64_t *ptrs[NR_DPUS];
                // point each slot to the right place in C_dpu
                for (unsigned i = 0; i < nr_of_dpus; ++i) {
                    ptrs[i] = (uint64_t*)(C_dpu + i * max_rows_per_dpu);
                }
                // dummy‐fill the rest
                for (unsigned i = nr_of_dpus; i < NR_DPUS; ++i) {
                    ptrs[i] = ptrs[0];
                }
                vud_simple_gather(&r,
                                  result_words,
                                  C_offset,
                                  &ptrs);
            }
		if(rep >= p.n_warmup)
			stop(&timer, 3);
	}
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
	printf("DPU-CPU Time (ms): ");
	print(&timer, 3, p.n_reps);
	printf("\n");

#if ENERGY
	printf("Energy (J): %f J\t", avg_energy);
#endif

	// Check output
	bool status = true;
	unsigned int n,j;
	i = 0;
	for (n = 0; n < nr_of_dpus; n++) {
		for (j = 0; j < dpu_info[n].rows_per_dpu; j++) {
		//for (j = 0; j < 5; j++) {
			if(C[i] != C_dpu[n * max_rows_per_dpu + j]) {
				status = false;
//#if PRINT
				printf("%d: %d -- %d\n", i, C[i], C_dpu[n * max_rows_per_dpu + j]);
//#endif
			break;
			}
			//printf("%d: %x -- %x\n", i, C[i], C_dpu[n * max_rows_per_dpu + j]);
			i++;
		}
	}
	if (status) {
		printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
	} else {
		printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
	}
#if 0
    uint64_t debug[64][128];
    uint64_t* ptrs[64];
    for(int i=0;i<64;i++) ptrs[i] = ptrs[i] = &debug[i][0];
    vud_simple_gather(&r, /*words=*/14, /*mr_offset=*/(64<<20)-128, &ptrs);
printf(
  "DPU0  args=(n=%llu,p=%llu,rows=%llu,max=%llu)\n"
  "      A[0..3]=(%llu,%llu,%llu,%llu)\n"
  "      B[0..3]=(%llu,%llu,%llu,%llu)\n"
  "      C_partial[0]=%llu\n"
  "      C_sum_partial[0]=%llu\n",
  debug[0][0], debug[0][1], debug[0][2], debug[0][3],
  debug[0][4], debug[0][5], debug[0][6], debug[0][7],
  debug[0][8], debug[0][9], debug[0][10], debug[0][11],
  debug[0][12], debug[0][13]);
#endif
#if 0
    uint64_t logs[64][SK_LOG_MAX_ENTRIES];
    int err = vud_log_read(&r, 64, logs);
    if (err) {
        printf("Log gather failed: %d\n", r.err);
    } else {
        for (int d = 0; d < 64; ++d) {
            printf("DPU %02d logs:", d);
            for (int i = 0; i < 2; ++i) {
                printf(" %llu", (unsigned long long)logs[d][i]);
            }
            printf("\n");
        }
    }
#endif

	// Deallocation
	free(A);
	free(B);
	free(C);
	free(C_dpu);
        vud_rank_free(&r);
        return 0;

#if ENERGY
	DPU_ASSERT(dpu_probe_deinit(&probe));
#endif
}
