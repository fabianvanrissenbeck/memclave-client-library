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
#include <time.h>
#include <inttypes.h>

#if ENERGY
#include <dpu_probe.h>
#endif

#define NR_DPUS 64

#include <vud.h>
#include <vud_sk.h>
#include <vud_ime.h>
#include <vud_mem.h>
#include "../../src/vud_log.h"
#include "support/common.h"    // defines T, dpu_arguments_t, dpu_results_t, etc.
#include "support/params.h"    // parses command-line into struct Params
#include "support/timer.h"     // for timing host vs DPU if you want

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "../mlp.sk"
#endif

/// total MRAM per DPU
#define MRAM_SIZE_BYTES     (64u << 20)
/// we reserve 64 B at the very top
#define SK_LOG_SIZE_BYTES   64
#define SK_LOG_OFFSET       (MRAM_SIZE_BYTES - SK_LOG_SIZE_BYTES)
static T* Z;

#define ARG_OFFSET     0x2000
#define ARG_SIZE       sizeof(dpu_arguments_t)
#define A_OFFSET       (ARG_OFFSET + ((ARG_SIZE + 0xFF) & ~0xFF))

// Use one "started" flag per bucket per rep to avoid re-zeroing on repeated starts
#define START_BUCKET(timer, idx, repidx, started_flag) \
    do { start((timer), (idx), (started_flag) ? 1 : (repidx)); (started_flag) = 1; } while (0)

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

// ---- tiny time helper ----
static inline double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// ---- push MRAM args (per-DPU array) ----
static void push_args_array(vud_rank* r, dpu_arguments_t* args, uint32_t nr_of_dpus) {
    const size_t words = (sizeof(dpu_arguments_t) + 7u) / 8u;   // expect 3
    //const uint64_t* p64[NR_DPUS];
    const dpu_arguments_t* p64[NR_DPUS];

    for (uint32_t i = 0; i < nr_of_dpus; ++i) p64[i] = (const uint64_t*)&args[i];
    for (uint32_t i = nr_of_dpus; i < NR_DPUS; ++i) p64[i] = p64[0];

    vud_simple_transfer(r, words, (const uint64_t (*)[NR_DPUS])&p64, ARG_OFFSET);
    if (r->err) { fprintf(stderr, "push_args_array: transfer failed\n"); }
}

// ---- gather 64B SK log (8 x u64) from all lanes ----
static void gather_sklog(vud_rank* r, uint64_t out_logs[64][8]) {
    uint64_t* ptrs[64];
    for (int d = 0; d < 64; ++d) ptrs[d] = &out_logs[d][0];
    vud_simple_gather(r, /*words=*/8, /*src=*/SK_LOG_OFFSET, &ptrs);
}

// ---- max over DPUs for a given slot ----
static inline uint64_t max_slot(uint64_t logs[64][8], int slot) {
    uint64_t mx = 0; for (int d = 0; d < 64; ++d) if (logs[d][slot] > mx) mx = logs[d][slot];
    return mx;
}

typedef struct { double f_hz; double baseline_ms; } dpu_calib_t;

static dpu_calib_t dpu_calibrate_old(vud_rank* r, dpu_arguments_t* args, uint32_t nr_of_dpus) {
    uint64_t logs[64][8];

    // A) baseline: no spin
    for (uint32_t i = 0; i < nr_of_dpus; ++i) args[i].spin_cycles = 1;
    //args[0].dummy = 0xabab;
    push_args_array(r, args, nr_of_dpus);

    double a0 = now_ms(); vud_ime_launch(r); vud_ime_wait(r); double a1 = now_ms();
    gather_sklog(r, logs);
    //printf("[sent/rcv args] args[0].dummy:%x/%x\n", args[0].dummy, logs[0][2]);
    printf("[baseline: no spin] dummy:%d counter_cycles:%llu, spin_cycles(sent/rcv):%llu/%llu\n", logs[0][5], logs[0][4], args[0].spin_cycles, logs[0][3]);
    uint64_t spinA = 0;  // should be ~0
    for (int d = 0; d < 64; ++d) if (logs[d][4] > spinA) spinA = logs[d][4];

    // B) long spin to dwarf host overhead (e.g., 50M cycles)
    for (uint32_t i = 0; i < nr_of_dpus; ++i) args[i].spin_cycles = 50ull * 1000ull * 1000ull;
    push_args_array(r, args, nr_of_dpus);

    double b0 = now_ms(); vud_ime_launch(r); vud_ime_wait(r); double b1 = now_ms();
    gather_sklog(r, logs);
    printf("[long spin] dummy:%d counter_cycles:%llu spin_cycles:%llu/%llu\n",logs[0][5], logs[0][4], args[0].spin_cycles, logs[0][3]);
    uint64_t spinB = 0;
    for (int d = 0; d < 64; ++d) if (logs[d][4] > spinB) spinB = logs[d][4];

    // Effective Hz for THIS run (subtract baseline ms to remove fixed overhead)
    double dt_ms = (b1 - b0) - (a1 - a0);
    if (dt_ms < 0.1) dt_ms = (b1 - b0);   // guard if baseline tiny
    double f_hz = (double)(spinB - spinA) / (dt_ms / 1000.0);
    printf("effective Hz:%f\n",f_hz);

    // Reset args (no spin) for real work, push once
    for (uint32_t i = 0; i < nr_of_dpus; ++i) args[i].spin_cycles = 0;
    push_args_array(r, args, nr_of_dpus);

    return (dpu_calib_t){ .f_hz = f_hz, .baseline_ms = (a1 - a0) };
}
static dpu_calib_t dpu_calibrate(vud_rank* r, dpu_arguments_t* args, uint32_t nr_of_dpus)
{
    // 1) Save real args, then zero sizes so the kernel does NO GEMV during calibration
    dpu_arguments_t *saved = malloc(nr_of_dpus * sizeof(*saved));
    memcpy(saved, args, nr_of_dpus * sizeof(*saved));
    for (uint32_t i = 0; i < nr_of_dpus; ++i) {
        args[i].n_size = 1;
        args[i].n_size_pad = 0;
        args[i].nr_rows = 0;
        args[i].max_rows = 0;
    }

    uint64_t logs[64][8];

    // A) Baseline: spin=0
    for (uint32_t i = 0; i < nr_of_dpus; ++i) args[i].spin_cycles = 10;
    push_args_array(r, args, nr_of_dpus);
    double a0 = now_ms(); vud_ime_launch(r); vud_ime_wait(r); double a1 = now_ms();
    gather_sklog(r, logs);
    printf("[baseline: no spin] t_spin1:%llu t_spin0:%llu, spin_cycles(sent/rcv):%llu/%llu\n", logs[0][5], logs[0][4], args[0].spin_cycles, logs[0][3]);
    uint64_t spinA = max_slot(logs, 4);  // (t_spin1 - t_spin0), should be ~0

    // B) Long spin
    for (uint32_t i = 0; i < nr_of_dpus; ++i) args[i].spin_cycles = 200ull * 1000ull * 1000ull;
    push_args_array(r, args, nr_of_dpus);
    double b0 = now_ms(); vud_ime_launch(r); vud_ime_wait(r); double b1 = now_ms();
    gather_sklog(r, logs);
    printf("[long spin] t_spin1:%llu t_spin0:%llu, spin_cycles:%llu/%llu\n",logs[0][5], logs[0][4], args[0].spin_cycles, logs[0][3]);
    uint64_t spinB = max_slot(logs, 4);  // ≈ 50,000,000

    // 2) Compute frequency (Hz) from pure spin delta
    double dt_ms = (b1 - b0) - (a1 - a0);
    if (dt_ms < 0.1) dt_ms = (b1 - b0);   // guard
    double f_hz = (double)(spinB - spinA) / (dt_ms / 1000.0);

    // 3) Restore real args (compute ON), spin=0
    memcpy(args, saved, nr_of_dpus * sizeof(*saved));
    for (uint32_t i = 0; i < nr_of_dpus; ++i) args[i].spin_cycles = 0;
    push_args_array(r, args, nr_of_dpus);
    free(saved);
    printf("[calib] f_hz: %.1f , baseline_ms: %.3f ms\n", f_hz, (a1 - a0));

    return (dpu_calib_t){ .f_hz = f_hz, .baseline_ms = (a1 - a0) };
}

// Main of the Host Application
int main(int argc, char **argv) {

	struct Params p = input_params(argc, argv);

	uint32_t nr_of_dpus = NR_DPUS;

	unsigned int i, l;
	unsigned int m_size = p.m_size;
	unsigned int n_size = p.n_size;

    	vud_rank r = vud_rank_alloc(VUD_ALLOC_ANY);
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
        vud_ime_load(&r, "../mlp");
 
        if (r.err) {
            puts("cannot load subkernel");
	    return -1;
        }
 
        uint8_t key[32];
        //random_key(key);
 
        //vud_ime_install_key(&r, key, NULL, NULL);
 
        if (r.err) {
            puts("key exchange failed");
	    return -1;
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

	T* B_batch = NULL;
        if (p.batch_size > 1) {
            if (posix_memalign((void**)&B_batch, 64, (size_t)p.batch_size * n_size * sizeof(T)) != 0) {
                fprintf(stderr, "B_batch alloc failed\n"); return EXIT_FAILURE;
            }
            // Fill deterministically; change as needed
            for (unsigned b = 0; b < p.batch_size; ++b)
                for (unsigned i = 0; i < n_size; ++i)
                    B_batch[(size_t)b*n_size + i] = (i + 13*b) & 1;
        }


	init_data(A, B, B_host, m_size, n_size);
	//printf("after init_data\n");

	// Compute output on CPU (performance comparison and verification purposes)
	start(&timer, 0, 0);
	mlp_host(C, A, B_host, m_size, n_size);
	stop(&timer, 0);
	//printf("after mlp_host\n");

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
	//dpu_calib_t calib = dpu_calibrate(&r, input_args, nr_of_dpus);
	dpu_calib_t calib = { .f_hz = 360025499, .baseline_ms = 5.2 };
        printf("[calib] DPU counter: %.1f MHz, baseline launch+wait: %.3f ms\n",
           calib.f_hz / 1e6, calib.baseline_ms);
#if 1
	//printf("before for loop\n");
	for (unsigned int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
		if (rep >= p.n_warmup)
			start(&timer, 1, rep - p.n_warmup);
	    int repidx = (int)(rep - p.n_warmup);
	    int s1=0, s2=0, s3=0, s4=0;
	    for (unsigned b = 0; b < p.batch_size; ++b) {
	       if (rep >= p.n_warmup) START_BUCKET(&timer, 1, repidx, s1);
	       //printf("before push dpu_arguments\n");
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


	       //printf("before scatter first layers weight matrix\n");
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
		
	       //printf("before broadcast input vec B\n");
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
	        if (rep >= p.n_warmup) START_BUCKET(&timer, 2, repidx, s2);

	        //printf("before launch sk\n");
		//DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
		//vud_ime_launch_sk(&r, DPU_BINARY);
		vud_ime_launch(&r);
                if (r.err) { 
                        printf(stderr,"launch failed\n"); return EXIT_FAILURE; 
                }
		if (rep >= p.n_warmup) stop(&timer, 2);
		
		double l0 = now_ms();
                vud_ime_wait(&r);
                if (r.err) { 
                        printf(stderr,"wait failed\n"); return EXIT_FAILURE; 
                }
		double l1 = now_ms();
		uint64_t logs[64][8];
                gather_sklog(&r, logs);
		// slot[1] = compute cycles (max over tasklets), written by DPU
		uint64_t compute_cycles = max_slot(logs, 1);
		double kernel_ms = (compute_cycles * 1000.0) / calib.f_hz;

		printf("DPU kernel (cycles→ms): %.3f ms  [cycles=%" PRIu64 ", f=%.1f MHz, host %.3f ms]\n",
		       kernel_ms, compute_cycles, calib.f_hz/1e6, (l1 - l0));


	       //printf("before lay for loop sk\n");
		for(int lay = 1; lay < NUM_LAYERS; lay++){
			if (rep >= p.n_warmup)
			        START_BUCKET(&timer, 4, repidx, s4);
			i = 0;

	       		//printf("before copy C_dpu\n");
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
			    size_t c_words = (max_rows_per_dpu * sizeof(T)) / 8;
                            vud_simple_gather(&r,
                                              c_words,  /* words per-DPU block */
                                              C_off,
                                              (uint64_t(*)[NR_DPUS])&ptrs);
                        }
	       		//printf("after copy C_dpu\n");

			// B = C
			unsigned int n, j;
			i = 0;
			for (n = 0; n < nr_of_dpus; n++) {
				for (j = 0; j < dpu_info[n].rows_per_dpu; j++) {
					B_tmp[i] = C_dpu[n * max_rows_per_dpu + j];
					i++;
				}
			}
		        //printf("before copy B_tmp\n");
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
                        
	                    //printf("before scater next layer weight matrix\n");
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
				START_BUCKET(&timer, 2, repidx, s2);
#if ENERGY
				DPU_ASSERT(dpu_probe_start(&probe));
#endif
			}

			//printf("before launch sk again\n");
			//DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
		double l0 = now_ms();
		        vud_ime_launch(&r);
                        if (r.err) { 
                                printf(stderr,"launch failed\n"); return EXIT_FAILURE; 
                        }
			if (rep >= p.n_warmup)
			{
				stop(&timer, 2);
#if ENERGY
				DPU_ASSERT(dpu_probe_stop(&probe));
#endif
			}
                        vud_ime_wait       (&r);
                        if (r.err) { 
                                printf(stderr,"wait failed\n"); return EXIT_FAILURE; 
                        }
		double l1 = now_ms();
		uint64_t logs[64][8];
                gather_sklog(&r, logs);
		// slot[1] = compute cycles (max over tasklets), written by DPU
		uint64_t compute_cycles = max_slot(logs, 1);
		double kernel_ms = (compute_cycles * 1000.0) / calib.f_hz;

		printf("DPU kernel (cycles→ms): %.3f ms  [cycles=%" PRIu64 ", f=%.1f MHz, host %.3f ms]\n",
		       kernel_ms, compute_cycles, calib.f_hz/1e6, (l1 - l0));

		}

#if PRINT
		// Display DPU Logs
		DPU_FOREACH(dpu_set, dpu) {
			DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
		}
#endif

			//printf("before gather\n");
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
		size_t c_words = (max_rows_per_dpu * sizeof(T)) / 8;
		START_BUCKET(&timer, 3, repidx, s3);
                vud_simple_gather(&r,
                                  c_words,     /* words per‐DPU block */
                                  C_off,
                                  (uint64_t(*)[NR_DPUS])&ptrs);
                }
		if(rep >= p.n_warmup)
			stop(&timer, 3);
	    }
	}
#endif

#if ENERGY
	double acc_energy, avg_energy, acc_time, avg_time;
	DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_ACCUMULATE, &acc_energy));
	DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &avg_energy));
	DPU_ASSERT(dpu_probe_get(&probe, DPU_TIME, DPU_ACCUMULATE, &acc_time));
	DPU_ASSERT(dpu_probe_get(&probe, DPU_TIME, DPU_AVERAGE, &avg_time));
#endif
	double total_us = (timer.time[1] + timer.time[2] + timer.time[4] + timer.time[3]); // CPU↔DPU + Kernel + Inter-DPU + DPU↔CPU
        double total_s  = total_us / 1e6;
        double thr      = ((double)p.batch_size) / total_s;
        printf("Throughput: %.2f samples/s  (batch=%u)\n", thr, p.batch_size);


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

#if ENERGY
	DPU_ASSERT(dpu_probe_deinit(&probe));
#endif
    
    
#if 0
#define LANES 64
#define GATHER_WORDS 1  // you're reading 1 x 8-byte word
static uint64_t logs[64];         // 1 word per possible DPU lane
static uint64_t sink[GATHER_WORDS] = {0};  // dummy sink for inactive lanes
static uint64_t* ptrs[64];
   for (int d = 0; d < nr_of_dpus; ++d) {
        //ptrs[d] = &logs[d];
	ptrs[d] = (d < 64) ? &logs[d * GATHER_WORDS] : &sink[0];
        //ptrs[d] = &logs[d];
    }
   for (int d = nr_of_dpus; d < NR_DPUS; ++d) {
        // cast bufferA to uint64_t* so we step in 8‑byte increments
        //ptrs[d] = (uint64_t*)((uint8_t *)bufferZ );
    }
    int err = 0;
    size_t nw = 1;                          // number of 8‑byte words
    vud_simple_gather(&r,
                  GATHER_WORDS,
                  SK_LOG_OFFSET,        // byte offset in MRAM
                  &ptrs);             // note the ‘&’ h
				      
    if (err) {
        printf("Log gather failed: %d\n", r.err);
    } else {
        for (int d = 0; d < NR_DPUS; ++d) {
            uint64_t v = (uint64_t)logs[d];
            printf("DPU %02d first element = %x\n", d, v);
        }
    }
#endif
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
        printf("DPU[%d] logs: %llu %llu %llu\n", d, magic, total, done);
    }
    printf("DPU cycles (whole-kernel, max over DPUs): %llu %llu %llu %llu\n",
           (unsigned long long) max_cycles, magic, tasks, T_BL);


	//return status ? 0 : -1;
        vud_rank_free(&r);
	return 0;
}
