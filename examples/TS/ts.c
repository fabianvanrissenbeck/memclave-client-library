/**
 * app.c
 * TS Host Application Source File
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <math.h>
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

#include "support/common.h"
#include "support/params.h"
#include "support/timer.h"
#include "support/prim_results.h"

#ifndef DPU_BINARY
#define DPU_BINARY "../ts"
#endif

#define MAX_DATA_VAL 127

#ifndef ARG_OFFSET
#define ARG_OFFSET   0x2000u
#endif
#define ARG_SIZE     ((uint32_t)sizeof(dpu_arguments_t))
#define A_OFFSET     (ARG_OFFSET + ((ARG_SIZE + 0xFFu) & ~0xFFu))

#define MAX_DATA_VAL 127

static DTYPE tSeries[1 << 26];
static DTYPE query  [1 << 15];
static DTYPE AMean  [1 << 26];
static DTYPE ASigma [1 << 26];
static DTYPE minHost;
static DTYPE minHostIdx;

// Create input arrays
static DTYPE *create_test_file(unsigned int ts_elements, unsigned int query_elements) {
	srand(0);

	for (uint64_t i = 0; i < ts_elements; i++)
	{
		tSeries[i] = i % MAX_DATA_VAL;
	}

	for (uint64_t i = 0; i < query_elements; i++)
	{
		query[i] = i % MAX_DATA_VAL;
	}

	return tSeries;
}

// Compute output in the host
static void streamp(DTYPE* tSeries, DTYPE* AMean, DTYPE* ASigma, int ProfileLength,
		DTYPE* query, int queryLength, DTYPE queryMean, DTYPE queryStdDeviation)
{
	DTYPE distance;
	DTYPE dotprod;
	minHost    = INT32_MAX;
	minHostIdx = 0;

	for (int subseq = 0; subseq < ProfileLength; subseq++)
	{
		dotprod = 0;
		for(int j = 0; j < queryLength; j++)
		{
			dotprod += tSeries[j + subseq] * query[j];
		}

		distance = 2 * (queryLength - (dotprod - queryLength * AMean[subseq]
					* queryMean) / (ASigma[subseq] * queryStdDeviation));

		if(distance < minHost)
		{
			minHost = distance;
			minHostIdx = subseq;
		}
	}
}

static void compute_ts_statistics(unsigned int timeSeriesLength, unsigned int ProfileLength, unsigned int queryLength)
{
	double* ACumSum = malloc(sizeof(double) * timeSeriesLength);
	ACumSum[0] = tSeries[0];
	for (uint64_t i = 1; i < timeSeriesLength; i++)
		ACumSum[i] = tSeries[i] + ACumSum[i - 1];
	double* ASqCumSum = malloc(sizeof(double) * timeSeriesLength);
	ASqCumSum[0] = tSeries[0] * tSeries[0];
	for (uint64_t i = 1; i < timeSeriesLength; i++)
		ASqCumSum[i] = tSeries[i] * tSeries[i] + ASqCumSum[i - 1];
	double* ASum = malloc(sizeof(double) * ProfileLength);
	ASum[0] = ACumSum[queryLength - 1];
	for (uint64_t i = 0; i < timeSeriesLength - queryLength; i++)
		ASum[i + 1] = ACumSum[queryLength + i] - ACumSum[i];
	double* ASumSq = malloc(sizeof(double) * ProfileLength);
	ASumSq[0] = ASqCumSum[queryLength - 1];
	for (uint64_t i = 0; i < timeSeriesLength - queryLength; i++)
		ASumSq[i + 1] = ASqCumSum[queryLength + i] - ASqCumSum[i];
	double * AMean_tmp = malloc(sizeof(double) * ProfileLength);
	for (uint64_t i = 0; i < ProfileLength; i++)
		AMean_tmp[i] = ASum[i] / queryLength;
	double* ASigmaSq = malloc(sizeof(double) * ProfileLength);
	for (uint64_t i = 0; i < ProfileLength; i++)
		ASigmaSq[i] = ASumSq[i] / queryLength - AMean[i] * AMean[i];
	for (uint64_t i = 0; i < ProfileLength; i++)
	{
		ASigma[i] = sqrt(ASigmaSq[i]);
		AMean[i]  = (DTYPE) AMean_tmp[i];
	}

	free(ACumSum);
	free(ASqCumSum);
	free(ASum);
	free(ASumSq);
	free(ASigmaSq);
	free(AMean_tmp);
}

static void push_args_array(vud_rank *r, dpu_arguments_t *args, uint32_t nr_of_dpus) {
    const vud_mram_size words = (vud_mram_size)((sizeof(dpu_arguments_t) + 7u) / 8u);
    _Alignas(8) uint64_t staged[NR_DPUS][8];  // up to 64B args (safe)
    assert(nr_of_dpus <= NR_DPUS && words <= 8);
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

	// Timer declaration
	Timer timer;

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

	unsigned long int ts_size =  p.input_size_n;
	const unsigned int query_length = p.input_size_m;

	// Size adjustment
	if(ts_size % (nr_of_dpus * NR_TASKLETS*query_length))
		ts_size = ts_size +  (nr_of_dpus * NR_TASKLETS * query_length - ts_size % (nr_of_dpus * NR_TASKLETS*query_length));

	// Create an input file with arbitrary data
	create_test_file(ts_size, query_length);
	compute_ts_statistics(ts_size, ts_size - query_length, query_length);

	DTYPE query_mean;
	double queryMean = 0;
	for(unsigned i = 0; i < query_length; i++) queryMean += query[i];
	queryMean /= (double) query_length;
	query_mean = (DTYPE) queryMean;

	DTYPE query_std;
	double queryStdDeviation;
	double queryVariance = 0;
	for(unsigned i = 0; i < query_length; i++)
	{
		queryVariance += (query[i] - queryMean) * (query[i] - queryMean);
	}
	queryVariance /= (double) query_length;
	queryStdDeviation = sqrt(queryVariance);
	query_std = (DTYPE) queryStdDeviation;

	DTYPE *bufferTS     = tSeries;
	DTYPE *bufferQ      = query;
	DTYPE *bufferAMean  = AMean;
	DTYPE *bufferASigma = ASigma;

	uint32_t slice_per_dpu = ts_size / nr_of_dpus;
        dpu_result_t result;
        result.minValue = INT32_MAX;
        result.minIndex = 0;
        result.maxValue = 0;
        result.maxIndex = 0;

	for (int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

		if (rep >= p.n_warmup)
			start(&timer, 1, rep - p.n_warmup);
                /* args */
                dpu_arguments_t args[NR_DPUS];
                for (uint32_t i = 0; i < nr_of_dpus; ++i) {
                    args[i].input_size_n     = (uint64_t)ts_size;
                    args[i].query_length     = (uint32_t)query_length;
                    args[i].query_mean       = (DTYPE)query_mean;
                    args[i].query_std        = (DTYPE)query_std;
                    args[i].slice_per_dpu    = (uint32_t)slice_per_dpu;
                    args[i].exclusion_zone   = 0;
                    args[i].kernel           = kernel1;
                }
                push_args_array(&r, args, nr_of_dpus);
                if (r.err) { fprintf(stderr, "args push failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
                
                /* MRAM layout offsets (must match subkernel) */
                const vud_mram_addr Q_OFF     = (vud_mram_addr)A_OFFSET;
                const size_t q_bytes          = (size_t)query_length * sizeof(DTYPE);
                
                const size_t ts_chunk_elems   = (size_t)slice_per_dpu + (size_t)query_length;
                const size_t ts_chunk_bytes   = ts_chunk_elems * sizeof(DTYPE);
                
                const vud_mram_addr TS_OFF    = (vud_mram_addr)(Q_OFF + (vud_mram_addr)q_bytes);
                const vud_mram_addr MEAN_OFF  = (vud_mram_addr)(TS_OFF + (vud_mram_addr)ts_chunk_bytes);
                const vud_mram_addr SIGMA_OFF = (vud_mram_addr)(MEAN_OFF + (vud_mram_addr)ts_chunk_bytes);
		const vud_mram_addr RESULTS_OFF = (vud_mram_addr)(SIGMA_OFF + (vud_mram_addr)ts_chunk_bytes);

                /* 1) push query (same to all DPUs) */
                {
                    const vud_mram_size wq = (vud_mram_size)((q_bytes + 7) / 8);
                    const uint64_t *lanes[NR_DPUS];
                    for (uint32_t d = 0; d < nr_of_dpus; ++d) lanes[d] = (const uint64_t*)query;
                    for (uint32_t d = nr_of_dpus; d < NR_DPUS;  ++d) lanes[d] = lanes[0];
                    vud_simple_transfer(&r, wq, (const uint64_t (*)[NR_DPUS])&lanes, Q_OFF);
                    if (r.err) { fprintf(stderr, "push query failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
                }
               
                /* 2) push TS slices (+overlap) */
                {
                    const vud_mram_size wt = (vud_mram_size)((ts_chunk_bytes + 7) / 8);
                    const uint64_t *lanes[NR_DPUS];
                    for (uint32_t d = 0; d < nr_of_dpus; ++d) lanes[d] = (const uint64_t*)(tSeries + (uint64_t)slice_per_dpu * d);
                    for (uint32_t d = nr_of_dpus; d < NR_DPUS;  ++d) lanes[d] = lanes[0];
                    vud_simple_transfer(&r, wt, (const uint64_t (*)[NR_DPUS])&lanes, TS_OFF);
                    if (r.err) { fprintf(stderr, "push TS failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
                }
               
                /* 3) push Mean slices */
                {
                    const vud_mram_size wm = (vud_mram_size)((ts_chunk_bytes + 7) / 8);
                    const uint64_t *lanes[NR_DPUS];
                    for (uint32_t d = 0; d < nr_of_dpus; ++d) lanes[d] = (const uint64_t*)(AMean + (uint64_t)slice_per_dpu * d);
                    for (uint32_t d = nr_of_dpus; d < NR_DPUS;  ++d) lanes[d] = lanes[0];
                    vud_simple_transfer(&r, wm, (const uint64_t (*)[NR_DPUS])&lanes, MEAN_OFF);
                    if (r.err) { fprintf(stderr, "push Mean failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
                }
               
                /* 4) push Sigma slices */
                {
                    const vud_mram_size ws = (vud_mram_size)((ts_chunk_bytes + 7) / 8);
                    const uint64_t *lanes[NR_DPUS];
                    for (uint32_t d = 0; d < nr_of_dpus; ++d) lanes[d] = (const uint64_t*)(ASigma + (uint64_t)slice_per_dpu * d);
                    for (uint32_t d = nr_of_dpus; d < NR_DPUS;  ++d) lanes[d] = lanes[0];
                    vud_simple_transfer(&r, ws, (const uint64_t (*)[NR_DPUS])&lanes, SIGMA_OFF);
                    if (r.err) { fprintf(stderr, "push Sigma failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }
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

                /* run */
                if (rep >= p.n_warmup) 
			start(&timer, 2, rep - p.n_warmup);
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

		if (rep >= p.n_warmup)
			start(&timer, 3, rep - p.n_warmup);
		/* gather NR_TASKLETS results per DPU from MRAM */
		const size_t res_bytes = (size_t)NR_TASKLETS * sizeof(dpu_result_t);
		const vud_mram_size wres = (vud_mram_size)((res_bytes + 7) / 8);

		uint64_t *tmp = (uint64_t*)malloc((size_t)wres * 8 * nr_of_dpus);
		if (!tmp) { fprintf(stderr, "tmp alloc failed\n"); return EXIT_FAILURE; }

		uint64_t *lanes[NR_DPUS];
		for (uint32_t d = 0; d < nr_of_dpus; ++d)
		    lanes[d] = &tmp[(size_t)d * wres];
		for (uint32_t d = nr_of_dpus; d < NR_DPUS; ++d)
		    lanes[d] = lanes[0];

		vud_simple_gather(&r, wres, RESULTS_OFF, (uint64_t* (*)[NR_DPUS])&lanes);
		if (r.err) { free(tmp); fprintf(stderr, "gather results failed: %s\n", vud_error_str(r.err)); return EXIT_FAILURE; }

		/* PRIM-style reduction on host */
		result.minValue = INT32_MAX;
		result.minIndex = 0;

		for (uint32_t d = 0; d < nr_of_dpus; ++d) {
		    dpu_result_t *rd = (dpu_result_t *)&tmp[(size_t)d * wres];  // packed results for this DPU
		    for (uint32_t t = 0; t < NR_TASKLETS; ++t) {
		        if (rd[t].minValue < result.minValue && rd[t].minValue > 0) {
		            result.minValue = rd[t].minValue;
		            result.minIndex = (DTYPE)rd[t].minIndex + (DTYPE)((uint64_t)d * (uint64_t)slice_per_dpu);
		        }
		    }
		}

		free(tmp);

		if(rep >= p.n_warmup)
			stop(&timer, 3);


#if PRINT
		printf("LOGS\n");
		DPU_FOREACH(dpu_set, dpu) {
			DPU_ASSERT(dpu_log_read(dpu, stdout));
		}
#endif

		if (rep >= p.n_warmup)
			start(&timer, 4, rep - p.n_warmup);
		streamp(tSeries, AMean, ASigma, ts_size - query_length - 1, query, query_length, query_mean, query_std);
		if(rep >= p.n_warmup)
			stop(&timer, 4);
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
	print(&timer, 4, p.n_reps);
	printf("Inter-DPU Time (ms): ");
	print(&timer, 0, p.n_reps);
	printf("CPU-DPU Time (ms): ");
	print(&timer, 1, p.n_reps);
	printf("DPU Kernel Time (ms): ");
	print(&timer, 2, p.n_reps);
	printf("DPU-CPU Time (ms): ");
	print(&timer, 3, p.n_reps);
    // update CSV
#define TEST_NAME "TS"
#define RESULTS_FILE "prim_results.csv"
    //update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 4, p.n_reps, "CPU");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 1, p.n_reps, "M_C2D");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 3, p.n_reps, "M_D2C");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 2, p.n_reps, "DPU");

#if ENERGY
	printf("Energy (J): %f J\t", avg_energy);
#endif

	int status = (minHost == result.minValue);
	if (status) {
		printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] results are equal\n");
	} else {
		printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] results differ!\n");
	}

        vud_rank_free(&r);

#if ENERGY
	DPU_ASSERT(dpu_probe_deinit(&probe));
#endif

	return 0;
}
