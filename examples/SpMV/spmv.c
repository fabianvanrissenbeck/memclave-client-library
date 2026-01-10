/**
 * spmv.c
 * SpMV Host Application Source File (Memclave/VUD port)
 */
#include <assert.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include <vud.h>
#include <vud_sk.h>
#include <vud_ime.h>
#include <vud_mem.h>
#include "../../src/vud_log.h"

#include "mram-management.h"
#include "support/common.h"
#include "support/matrix.h"
#include "support/params.h"
#include "support/timer.h"
#include "support/utils.h"
#include "support/prim_results.h"

#define NR_DPUS        64
#define NR_TASKLETS    16

#define ARG_OFFSET     0x3000
#define DPU_BINARY     "../spmv"

#ifndef ENERGY
#define ENERGY 0
#endif

/// total MRAM per DPU
#define MRAM_SIZE_BYTES     (64u << 20)
/// reserve 64B at the very top (used by some ports for SK log)
#define SK_LOG_SIZE_BYTES   64
#define SK_LOG_OFFSET       (MRAM_SIZE_BYTES - SK_LOG_SIZE_BYTES)

#define ALIGN256(x)         (((x) + 0xFFu) & ~0xFFu)

static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static inline uint32_t umin_u32(uint32_t a, uint32_t b) {
    return (a < b) ? a : b;
}

int main(int argc, char** argv) {

    // Process parameters
    struct Params p = input_params(argc, argv);

    // Timing and profiling
    Timer timer;
    float loadTime = 0.0f, dpuTime = 0.0f, retrieveTime = 0.0f;

    // Allocate VUD rank + load subkernel
    const uint32_t numDPUs = NR_DPUS;
    vud_rank r = vud_rank_alloc(VUD_ALLOC_ANY);
    if (r.err) { fprintf(stderr, "vud_rank_alloc failed\n"); return EXIT_FAILURE; }

    vud_ime_wait(&r);
    if (r.err) { fprintf(stderr, "vud_ime_wait failed\n"); return EXIT_FAILURE; }

    vud_ime_load(&r, DPU_BINARY);
    if (r.err) { fprintf(stderr, "cannot load subkernel '%s'\n", DPU_BINARY); return EXIT_FAILURE; }

    PRINT_INFO(p.verbosity >= 1, "Allocated %u DPU(s)", numDPUs);

    // Initialize SpMV data structures
    PRINT_INFO(p.verbosity >= 1, "Reading matrix %s", p.fileName);
    struct COOMatrix cooMatrix = readCOOMatrix(p.fileName);
    PRINT_INFO(p.verbosity >= 1, "    %u rows, %u columns, %u nonzeros",
               cooMatrix.numRows, cooMatrix.numCols, cooMatrix.numNonzeros);

    struct CSRMatrix csrMatrix = coo2csr(cooMatrix);
    const uint32_t numRows = csrMatrix.numRows;
    const uint32_t numCols = csrMatrix.numCols;
    uint32_t* rowPtrs = csrMatrix.rowPtrs;
    struct Nonzero* nonzeros = csrMatrix.nonzeros;

    float* inVector  = (float*)malloc(ROUND_UP_TO_MULTIPLE_OF_8(numCols * sizeof(float)));
    float* outVector = (float*)malloc(ROUND_UP_TO_MULTIPLE_OF_8(numRows * sizeof(float)));
    if (!inVector || !outVector) { fprintf(stderr, "malloc vectors failed\n"); return EXIT_FAILURE; }
    initVector(inVector, numCols);
    memset(outVector, 0, ROUND_UP_TO_MULTIPLE_OF_8(numRows * sizeof(float)));

    // Partition rows across DPUs
    const uint32_t numRowsPerDPU = ROUND_UP_TO_MULTIPLE_OF_2((numRows - 1) / numDPUs + 1);
    PRINT_INFO(p.verbosity >= 1, "Assigning %u rows per DPU", numRowsPerDPU);

    // Compute max NNZ per partition to allocate UNIFORM MRAM buffers
    uint32_t maxNNZ = 0;
    for (uint32_t d = 0; d < numDPUs; ++d) {
        uint32_t start = d * numRowsPerDPU;
        if (start >= numRows) break;
        uint32_t end = umin_u32(start + numRowsPerDPU, numRows);
        uint32_t nnz = rowPtrs[end] - rowPtrs[start];
        if (nnz > maxNNZ) maxNNZ = nnz;
    }

    // Uniform sizes (pad to 8B)
    const uint32_t ROWPTR_BYTES     = ROUND_UP_TO_MULTIPLE_OF_8((numRowsPerDPU + 1) * sizeof(uint32_t));
    // +1 element to tolerate the "one extra unused seqread_get()" beyond last NNZ
    const uint32_t NONZEROS_BYTES   = ROUND_UP_TO_MULTIPLE_OF_8((maxNNZ + 1) * sizeof(struct Nonzero));
    const uint32_t INVEC_BYTES      = ROUND_UP_TO_MULTIPLE_OF_8(numCols * sizeof(float));
    const uint32_t OUTVEC_BYTES_DPU = ROUND_UP_TO_MULTIPLE_OF_8(numRowsPerDPU * sizeof(float)); // multiple-of-8 by construction

    PRINT_INFO(p.verbosity >= 1,
        "Uniform sizes (bytes): rowptr=%u, nonzeros=%u, invec=%u, outvec_per_dpu=%u",
        ROWPTR_BYTES, NONZEROS_BYTES, INVEC_BYTES, OUTVEC_BYTES_DPU);

    // Uniform MRAM layout (same across all DPUs)
    const uint32_t HEAP_BASE     = ARG_OFFSET + ALIGN256(sizeof(struct DPUParams));
    const uint32_t dpuRowPtrs_m  = HEAP_BASE;
    const uint32_t dpuNonzeros_m = dpuRowPtrs_m  + ROWPTR_BYTES;
    const uint32_t dpuInVec_m    = dpuNonzeros_m + NONZEROS_BYTES;
    const uint32_t dpuOutVec_m   = dpuInVec_m    + INVEC_BYTES;
    const uint32_t totalBytes    = dpuOutVec_m   + OUTVEC_BYTES_DPU;

    // Per-DPU params + per-DPU host buffers (must exist for all 64 DPUs)
    struct DPUParams dpuParams[NR_DPUS];
    void* rowptr_bufs[NR_DPUS];
    void* nz_bufs[NR_DPUS];

    for (uint32_t d = 0; d < numDPUs; ++d) {

        uint32_t dpuStartRowIdx = d * numRowsPerDPU;

        // Determine how many rows this DPU owns
        uint32_t dpuNumRows = 0;
        if (dpuStartRowIdx < numRows) {
            uint32_t end = umin_u32(dpuStartRowIdx + numRowsPerDPU, numRows);
            dpuNumRows = end - dpuStartRowIdx;
        }

        dpuParams[d].dpuNumRows = dpuNumRows;

        // RowPtrs offset (safe even when dpuNumRows==0)
        uint32_t rowPtrBase = rowPtrs[(dpuStartRowIdx < numRows) ? dpuStartRowIdx : numRows];
        dpuParams[d].dpuRowPtrsOffset = rowPtrBase;

        // MRAM pointers/offsets
        dpuParams[d].dpuRowPtrs_m   = dpuRowPtrs_m;
        dpuParams[d].dpuNonzeros_m  = dpuNonzeros_m;
        dpuParams[d].dpuInVector_m  = dpuInVec_m;
        dpuParams[d].dpuOutVector_m = dpuOutVec_m;

        // --- Build padded rowPtrs slice ---
        uint32_t* rp = (uint32_t*)malloc(ROWPTR_BYTES);
        if (!rp) { fprintf(stderr, "malloc rowptr slice failed\n"); exit(1); }
        rowptr_bufs[d] = rp;

        if (dpuNumRows > 0) {
            uint32_t slice_len = dpuNumRows + 1;
            memcpy(rp, &rowPtrs[dpuStartRowIdx], slice_len * sizeof(uint32_t));
            uint32_t last = rp[slice_len - 1];
            for (uint32_t i = slice_len; i < ROWPTR_BYTES / sizeof(uint32_t); ++i) rp[i] = last;
        } else {
            // No rows: repeat terminal rowPtr
            uint32_t last = rowPtrs[numRows];
            for (uint32_t i = 0; i < ROWPTR_BYTES / sizeof(uint32_t); ++i) rp[i] = last;
        }

        // --- Build padded nonzeros slice ---
        struct Nonzero* nz = (struct Nonzero*)malloc(NONZEROS_BYTES);
        if (!nz) { fprintf(stderr, "malloc nonzeros slice failed\n"); exit(1); }
        nz_bufs[d] = nz;
        memset(nz, 0, NONZEROS_BYTES);

        if (dpuNumRows > 0) {
            uint32_t start = dpuStartRowIdx;
            uint32_t end   = umin_u32(start + dpuNumRows, numRows);

            uint32_t dpuRowPtrsOffset = rowPtrs[start];
            uint32_t dpuNumNonzeros   = rowPtrs[end] - dpuRowPtrsOffset;

            // Copy partition nonzeros into the front of the padded buffer
            memcpy(nz, &nonzeros[dpuRowPtrsOffset], dpuNumNonzeros * sizeof(struct Nonzero));
            // Remaining space stays zero-padded (and we have +1 element slack).
        }
    }

    PRINT_INFO(p.verbosity == 1, "Copying data to DPUs");

    // Scatter params
    startTimer(&timer);
    {
        const void* param_slices[NR_DPUS];
        for (uint32_t d = 0; d < numDPUs; ++d) param_slices[d] = &dpuParams[d];

        scatterToDPU(&r, (const void*(*)[NR_DPUS])&param_slices, ARG_OFFSET, sizeof(struct DPUParams));
    }
    stopTimer(&timer); loadTime += getElapsedTime(timer);

    // Scatter rowPtrs + nonzeros
    startTimer(&timer);
    scatterToDPU(&r, (const void*(*)[NR_DPUS])&rowptr_bufs, dpuRowPtrs_m, ROWPTR_BYTES);
    scatterToDPU(&r, (const void*(*)[NR_DPUS])&nz_bufs,     dpuNonzeros_m, NONZEROS_BYTES);
    stopTimer(&timer); loadTime += getElapsedTime(timer);

    // Broadcast input vector (same to all DPUs)
    startTimer(&timer);
    copyToDPU(&r, inVector, dpuInVec_m, INVEC_BYTES);
    stopTimer(&timer); loadTime += getElapsedTime(timer);

    // Free host scatter buffers
    for (uint32_t d = 0; d < numDPUs; ++d) {
        free(rowptr_bufs[d]);
        free(nz_bufs[d]);
    }

    PRINT_INFO(p.verbosity >= 1, "    CPU-DPU Time: %f ms", loadTime * 1e3);

    // Run all DPUs
    PRINT_INFO(p.verbosity >= 1, "Booting DPUs");
    //double t0 = now_ms();
    startTimer(&timer);
    vud_ime_launch(&r);
    if (r.err) { fprintf(stderr, "vud_ime_launch failed %d\n", r.err); return EXIT_FAILURE; }
    vud_ime_wait(&r);
    if (r.err) { fprintf(stderr, "vud_ime_wait failed %d\n", r.err); return EXIT_FAILURE; }
    //double t1 = now_ms();
    stopTimer(&timer);
    dpuTime += getElapsedTime(timer); ;
    PRINT_INFO(p.verbosity >= 1, "    DPU Time: %f ms", dpuTime * 1e3);

    // Copy back result
    PRINT_INFO(p.verbosity >= 1, "Copying back the result");
    startTimer(&timer);
    {
        uint8_t* out_all = (uint8_t*)malloc((size_t)numDPUs * OUTVEC_BYTES_DPU);
        if (!out_all) { fprintf(stderr, "malloc out_all failed\n"); exit(1); }

        void* out_ptrs[NR_DPUS];
        for (uint32_t d = 0; d < numDPUs; ++d)
            out_ptrs[d] = out_all + (size_t)d * OUTVEC_BYTES_DPU;

        gatherFromDPU(&r, dpuOutVec_m, (void*(*)[NR_DPUS])&out_ptrs, OUTVEC_BYTES_DPU);

        // Stitch back into outVector
        for (uint32_t d = 0; d < numDPUs; ++d) {
            uint32_t dpuStartRowIdx = d * numRowsPerDPU;
            uint32_t dpuNumRows = dpuParams[d].dpuNumRows;
            if (dpuNumRows == 0 || dpuStartRowIdx >= numRows) continue;

            memcpy(outVector + dpuStartRowIdx,
                   out_all + (size_t)d * OUTVEC_BYTES_DPU,
                   (size_t)dpuNumRows * sizeof(float));
        }

        free(out_all);
    }
    stopTimer(&timer);
    retrieveTime += getElapsedTime(timer);

    PRINT_INFO(p.verbosity >= 1, "    DPU-CPU Time: %f ms", retrieveTime * 1e3);

    // CPU reference
    PRINT_INFO(p.verbosity >= 1, "Calculating result on CPU");
    startTimer(&timer);
    float* outVectorReference = (float*)malloc(numRows * sizeof(float));
    for (uint32_t rowIdx = 0; rowIdx < numRows; ++rowIdx) {
        float sum = 0.0f;
        for (uint32_t i = rowPtrs[rowIdx]; i < rowPtrs[rowIdx + 1]; ++i) {
            uint32_t colIdx = nonzeros[i].col;
            float value = nonzeros[i].value;
            sum += inVector[colIdx] * value;
        }
        outVectorReference[rowIdx] = sum;
    }
    stopTimer(&timer);
    float cpuTime = getElapsedTime(timer);
    if (p.verbosity >= 0) {
        PRINT("CPU Time(ms): %f, CPU-DPU Time(ms): %f    DPU Kernel Time (ms): %f    DPU-CPU Time (ms): %f",
              cpuTime * 1e3, loadTime * 1e3, dpuTime * 1e3, retrieveTime * 1e3);
    }
        // update CSV
#define TEST_NAME "SpMV"
#define RESULTS_FILE "prim_results.csv"
        //update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 0, hostTime, "CPU");
        update_csv(RESULTS_FILE, TEST_NAME, "M_C2D", loadTime*1e3);
        update_csv(RESULTS_FILE, TEST_NAME, "M_D2C", retrieveTime*1e3);
        update_csv(RESULTS_FILE, TEST_NAME, "DPU", dpuTime*1e3);

    // Verify
    PRINT_INFO(p.verbosity >= 1, "Verifying the result");
    const float tolerance = 0.00001f;
    int i = 0;
    bool status = true;
    for (uint32_t rowIdx = 0; rowIdx < numRows; ++rowIdx) {
        float ref = outVectorReference[rowIdx];
        float got = outVector[rowIdx];

        float diff;
        if (ref == 0.0f) diff = got;               // absolute
        else             diff = (ref - got) / ref; // relative

        if (diff > tolerance || diff < -tolerance) {
            PRINT_ERROR("Mismatch at index %u (CPU result = %f, DPU result = %f)", rowIdx, ref, got);
	    status = false;
	    i++;
	    if (i>10)
		break;
        }
    }
    if (status) {
        printf("\n[OK] Outputs are equal\n");
    } else {
        printf("\n[ERROR] Outputs differ!\n");
    }

    // Cleanup
    freeCOOMatrix(cooMatrix);
    freeCSRMatrix(csrMatrix);
    free(inVector);
    free(outVector);
    free(outVectorReference);

    vud_rank_free(&r);
    return 0;
}
