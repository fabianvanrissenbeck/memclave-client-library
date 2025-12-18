/**
* app.c
* BFS Host Application Source File
*
*/
#include <assert.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <inttypes.h>

#include <vud.h>
#include <vud_sk.h>
#include <vud_ime.h>
#include <vud_mem.h>
#include "../../src/vud_log.h"
#include "mram-management.h"
#include "support/common.h"
#include "support/graph.h"
#include "support/params.h"
#include "support/timer.h"
#include "support/utils.h"
#include "support/prim_results.h"

#define TASKLETS       16
#define NR_DPUS        64
#define ARG_OFFSET     0x3000
#define DPU_BINARY     "../bfs"

#ifndef ENERGY
#define ENERGY 0
#endif

/// total MRAM per DPU
#define MRAM_SIZE_BYTES     (64u << 20)
/// reserve 64B at the very top
#define SK_LOG_SIZE_BYTES   64
#define SK_LOG_OFFSET       (MRAM_SIZE_BYTES - SK_LOG_SIZE_BYTES)

#define ALIGN256(x)         (((x) + 0xFFu) & ~0xFFu)

static inline double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void gather_sklog(vud_rank* r, uint64_t out_logs[64][8]) {
    uint64_t* ptrs[64];
    for (int d = 0; d < 64; ++d) ptrs[d] = &out_logs[d][0];
    vud_simple_gather(r, /*words=*/8, /*src=*/SK_LOG_OFFSET, &ptrs);
}

static inline uint64_t max_slot(uint64_t logs[64][8], int slot) {
    uint64_t mx = 0;
    for (int d = 0; d < 64; ++d) if (logs[d][slot] > mx) mx = logs[d][slot];
    return mx;
}

typedef struct { double f_hz; double baseline_ms; } dpu_calib_t;

int main(int argc, char** argv) {

    struct Params p = input_params(argc, argv);

    // Fixed calibration (as in your other ports)
    dpu_calib_t calib = { .f_hz = 360025499, .baseline_ms = 5.2 };
    PRINT_INFO(p.verbosity >= 1,
        "[calib] DPU counter: %.1f MHz, baseline launch+wait: %.3f ms",
        calib.f_hz / 1e6, calib.baseline_ms);

    Timer timer;
    float loadTime = 0.0f, dpuTime = 0.0f, hostTime = 0.0f, retrieveTime = 0.0f;

    // Allocate VUD rank + load SK
    const uint32_t numDPUs = NR_DPUS;
    vud_rank r = vud_rank_alloc(VUD_ALLOC_ANY);
    if (r.err) { fprintf(stderr, "vud_rank_alloc failed\n"); return EXIT_FAILURE; }

    vud_ime_wait(&r);
    if (r.err) { fprintf(stderr, "vud_ime_wait failed\n"); return EXIT_FAILURE; }

    vud_ime_load(&r, DPU_BINARY);
    if (r.err) { fprintf(stderr, "cannot load subkernel\n"); return EXIT_FAILURE; }

    PRINT_INFO(p.verbosity >= 1, "Allocated %u DPU(s)", numDPUs);

    // Read graph + CSR
    PRINT_INFO(p.verbosity >= 1, "Reading graph %s", p.fileName);
    struct COOGraph cooGraph = readCOOGraph(p.fileName);
    PRINT_INFO(p.verbosity >= 1, "    Graph has %u nodes and %u edges", cooGraph.numNodes, cooGraph.numEdges);
    struct CSRGraph csrGraph = coo2csr(cooGraph);

    const uint32_t numNodes = csrGraph.numNodes;
    uint32_t* nodePtrs = csrGraph.nodePtrs;
    uint32_t* neighborIdxs = csrGraph.neighborIdxs;

    uint32_t* nodeLevel = calloc(numNodes, sizeof(uint32_t));
    uint64_t* visited = calloc(numNodes/64, sizeof(uint64_t));
    uint64_t* currentFrontier = calloc(numNodes/64, sizeof(uint64_t));
    uint64_t* nextFrontier = calloc(numNodes/64, sizeof(uint64_t));

    // init frontier to node 0 (same as PRIM)
    setBit(nextFrontier[0], 0);
    uint32_t level = 1;

    // Partitioning
    const uint32_t numNodesPerDPU = ROUND_UP_TO_MULTIPLE_OF_64((numNodes - 1)/numDPUs + 1);
    PRINT_INFO(p.verbosity >= 1, "Assigning %u nodes per DPU", numNodesPerDPU);

    // --- uniform MRAM sizes ---
    const uint32_t ROWPTR_BYTES    = ROUND_UP_TO_MULTIPLE_OF_8((numNodesPerDPU + 1) * sizeof(uint32_t));

    // max neighbors across partitions
    uint32_t maxNbrs = 0;
    for (unsigned d = 0; d < numDPUs; ++d) {
        uint32_t start = d * numNodesPerDPU;
        if (start >= numNodes) break;
        uint32_t end = start + numNodesPerDPU;
        if (end > numNodes) end = numNodes;
        uint32_t off  = nodePtrs[start];
        uint32_t nbrs = nodePtrs[end] - off;
        if (nbrs > maxNbrs) maxNbrs = nbrs;
    }

    const uint32_t COLIDX_BYTES    = ROUND_UP_TO_MULTIPLE_OF_8(maxNbrs * sizeof(uint32_t));
    const uint32_t NODELEVEL_BYTES = ROUND_UP_TO_MULTIPLE_OF_8(numNodesPerDPU * sizeof(uint32_t));
    const uint32_t GLOBAL_BM_BYTES = ROUND_UP_TO_MULTIPLE_OF_8((numNodes/64) * sizeof(uint64_t));
    const uint32_t CUR_BM_BYTES    = ROUND_UP_TO_MULTIPLE_OF_8((numNodesPerDPU/64) * sizeof(uint64_t));

    PRINT_INFO(p.verbosity >= 1,
        "Uniform sizes (bytes): rowptr=%u, colidx=%u, nodelevel=%u, visited=%u, cur_frontier=%u, next_frontier=%u, next_priv=%u",
        ROWPTR_BYTES, COLIDX_BYTES, NODELEVEL_BYTES, GLOBAL_BM_BYTES, CUR_BM_BYTES, GLOBAL_BM_BYTES,
        (uint32_t)(GLOBAL_BM_BYTES * TASKLETS));

    // --- uniform MRAM layout ---
    const uint32_t HEAP_BASE      = ARG_OFFSET + ALIGN256(sizeof(struct DPUParams));
    const uint32_t dpuNodePtrs_m  = HEAP_BASE;
    const uint32_t dpuNbrs_m      = dpuNodePtrs_m + ROWPTR_BYTES;
    const uint32_t dpuLevel_m     = dpuNbrs_m     + COLIDX_BYTES;
    const uint32_t dpuVisited_m   = dpuLevel_m    + NODELEVEL_BYTES;
    const uint32_t dpuCur_m       = dpuVisited_m  + GLOBAL_BM_BYTES;
    const uint32_t dpuNext_m      = dpuCur_m      + CUR_BM_BYTES;
    const uint32_t dpuNextPriv_m  = dpuNext_m     + GLOBAL_BM_BYTES;
    const uint32_t totalBytes     = dpuNextPriv_m + GLOBAL_BM_BYTES * TASKLETS;

    // Make sure we never collide with the SK log area
    assert(totalBytes <= SK_LOG_OFFSET);

    // Build per-DPU params and per-DPU host buffers (VALID for ALL 64 lanes)
    struct DPUParams dpuParams[NR_DPUS];
    void* rowptr_bufs[NR_DPUS];
    void* nbr_bufs[NR_DPUS];
    void* level_bufs[NR_DPUS];

    for (unsigned d = 0; d < numDPUs; ++d) {
        uint32_t dpuStartNodeIdx = d * numNodesPerDPU;

        // handle dpus beyond the end safely
        uint32_t dpuNumNodes = 0;
        if (dpuStartNodeIdx < numNodes) {
            uint32_t end = dpuStartNodeIdx + numNodesPerDPU;
            if (end > numNodes) end = numNodes;
            dpuNumNodes = end - dpuStartNodeIdx;
        } else {
            dpuStartNodeIdx = numNodes; // clamp
            dpuNumNodes = 0;
        }

        // --- params ---
        dpuParams[d].numNodes              = numNodes;
        dpuParams[d].dpuStartNodeIdx       = d * numNodesPerDPU;   // keep original meaning
        dpuParams[d].dpuNumNodes           = dpuNumNodes;
        dpuParams[d].dpuNodePtrsOffset     = (d * numNodesPerDPU < numNodes) ? nodePtrs[d * numNodesPerDPU] : nodePtrs[numNodes];
        dpuParams[d].level                 = level;

        dpuParams[d].dpuNodePtrs_m         = dpuNodePtrs_m;
        dpuParams[d].dpuNeighborIdxs_m     = dpuNbrs_m;
        dpuParams[d].dpuNodeLevel_m        = dpuLevel_m;
        dpuParams[d].dpuVisited_m          = dpuVisited_m;
        dpuParams[d].dpuCurrentFrontier_m  = dpuCur_m;
        dpuParams[d].dpuNextFrontier_m     = dpuNext_m;
        //dpuParams[d].dpuNextFrontierPriv_m = dpuNextPriv_m;

        // --- rowptr buffer (always allocated) ---
        uint32_t* rp = (uint32_t*)malloc(ROWPTR_BYTES);
        rowptr_bufs[d] = rp;

        if (d * numNodesPerDPU < numNodes) {
            uint32_t start = d * numNodesPerDPU;
            uint32_t end   = start + dpuNumNodes;
            uint32_t slice_len = dpuNumNodes + 1;
            memcpy(rp, &nodePtrs[start], slice_len * sizeof(uint32_t));
            uint32_t last = rp[slice_len - 1];
            for (uint32_t i = slice_len; i < ROWPTR_BYTES / sizeof(uint32_t); ++i) rp[i] = last;
        } else {
            // no nodes: repeat terminal pointer
            uint32_t last = nodePtrs[numNodes];
            for (uint32_t i = 0; i < ROWPTR_BYTES / sizeof(uint32_t); ++i) rp[i] = last;
        }

        // --- neighbor buffer (always allocated) ---
        uint32_t* nb = (uint32_t*)malloc(COLIDX_BYTES);
        nbr_bufs[d] = nb;
        memset(nb, 0, COLIDX_BYTES);

        if (dpuNumNodes > 0) {
            uint32_t start = d * numNodesPerDPU;
            uint32_t end   = start + dpuNumNodes;
            uint32_t off   = nodePtrs[start];
            uint32_t nbrs  = nodePtrs[end] - off;
            memcpy(nb, neighborIdxs + off, nbrs * sizeof(uint32_t));
        }

        // --- level buffer (always allocated) ---
        uint32_t* lv = (uint32_t*)malloc(NODELEVEL_BYTES);
        level_bufs[d] = lv;
        memset(lv, 0, NODELEVEL_BYTES); // initial level=0
    }

    // --- Scatter params once ---
    startTimer(&timer);
    {
        const void* param_slices[NR_DPUS];
        for (unsigned d = 0; d < numDPUs; ++d) param_slices[d] = &dpuParams[d];
        scatterToDPU(&r, (const void*(*)[NR_DPUS])&param_slices, ARG_OFFSET, sizeof(struct DPUParams));
    }
    stopTimer(&timer); loadTime += getElapsedTime(timer);

    // --- Scatter CSR + nodeLevel once ---
    startTimer(&timer);
    scatterToDPU(&r, (const void*(*)[NR_DPUS])&rowptr_bufs, dpuNodePtrs_m, ROWPTR_BYTES);
    scatterToDPU(&r, (const void*(*)[NR_DPUS])&nbr_bufs,    dpuNbrs_m,     COLIDX_BYTES);
    scatterToDPU(&r, (const void*(*)[NR_DPUS])&level_bufs,  dpuLevel_m,    NODELEVEL_BYTES);
    stopTimer(&timer); loadTime += getElapsedTime(timer);

    // free host scatter buffers now
    for (unsigned d = 0; d < numDPUs; ++d) {
        free(rowptr_bufs[d]);
        free(nbr_bufs[d]);
        free(level_bufs[d]);
    }

    // --- broadcast visited + initial nextFrontier once ---
    startTimer(&timer);
    copyToDPU(&r, visited,      dpuVisited_m, GLOBAL_BM_BYTES);
    copyToDPU(&r, nextFrontier, dpuNext_m,    GLOBAL_BM_BYTES);
    stopTimer(&timer); loadTime += getElapsedTime(timer);

    PRINT_INFO(p.verbosity >= 1, "    CPU-DPU Time: %f ms", loadTime * 1e3);

    // BFS loop
    uint32_t nextFrontierEmpty = 0;
    const uint32_t numTiles = numNodes / 64;

    while (!nextFrontierEmpty) {

        PRINT_INFO(p.verbosity >= 1, "Processing current frontier for level %u (distance %u)", level, level - 1);

        // Launch kernel
        //double l0 = now_ms();
	startTimer(&timer);
        vud_ime_launch(&r);
        if (r.err) { fprintf(stderr, "vud_ime_launch failed %d\n", r.err); return EXIT_FAILURE; }
        vud_ime_wait(&r);
        if (r.err) { fprintf(stderr, "vud_ime_wait failed %d\n", r.err); return EXIT_FAILURE; }
        //double l1 = now_ms();
        stopTimer(&timer);
        dpuTime += getElapsedTime(timer);
        PRINT_INFO(p.verbosity >= 2, "    Level DPU Time: %f ms", getElapsedTime(timer)*1e3);

        // Use SK log for kernel cycles
        //uint64_t logs[64][8];
        //gather_sklog(&r, logs);
        //uint64_t compute_cycles = max_slot(logs, 1);
        //double kernel_ms = (compute_cycles * 1000.0) / calib.f_hz;
        //dpuTime += kernel_ms;

        //PRINT_INFO(p.verbosity >= 2,
        //    "DPU kernel (cycles→ms): %.3f ms [cycles=%" PRIu64 ", f=%.1f MHz, host %.3f ms]",
        //    kernel_ms, compute_cycles, calib.f_hz/1e6, getElapsedTime(timer)*1e3);

        // Gather nextFrontier from ALL DPUs and union them
        startTimer(&timer);

        memset(currentFrontier, 0, (size_t)numTiles * sizeof(uint64_t));

        const size_t frontierBytes = (size_t)numTiles * sizeof(uint64_t);
        uint8_t* allFrontiers = (uint8_t*)malloc((size_t)numDPUs * frontierBytes);
        if (!allFrontiers) { fprintf(stderr, "malloc(allFrontiers) failed\n"); exit(1); }

        void* frontier_ptrs[NR_DPUS];
        for (unsigned d = 0; d < numDPUs; ++d)
            frontier_ptrs[d] = allFrontiers + (size_t)d * frontierBytes;

        gatherFromDPU(&r,
                      dpuNext_m,
                      (void*(*)[NR_DPUS])&frontier_ptrs,
                      frontierBytes);

        for (unsigned d = 0; d < numDPUs; ++d) {
            uint64_t* thisF = (uint64_t*)(allFrontiers + (size_t)d * frontierBytes);
            for (uint32_t t = 0; t < numTiles; ++t) currentFrontier[t] |= thisF[t];
        }

        free(allFrontiers);

        // Check empty
        nextFrontierEmpty = 1;
        for (uint32_t t = 0; t < numTiles; ++t) {
            if (currentFrontier[t]) { nextFrontierEmpty = 0; break; }
        }

        if (!nextFrontierEmpty) {
            ++level;

            // broadcast the new frontier as nextFrontier input
            copyToDPU(&r, currentFrontier, dpuNext_m, frontierBytes);

            // update level in params and re-scatter params
            for (unsigned d = 0; d < numDPUs; ++d) dpuParams[d].level = level;

            const void* param_slices[NR_DPUS];
            for (unsigned d = 0; d < numDPUs; ++d) param_slices[d] = &dpuParams[d];

            scatterToDPU(&r,
                         (const void*(*)[NR_DPUS])&param_slices,
                         ARG_OFFSET,
                         sizeof(struct DPUParams));
        }

        stopTimer(&timer);
        hostTime += getElapsedTime(timer);
        PRINT_INFO(p.verbosity >= 2, "    Level Inter-DPU Time: %f ms", getElapsedTime(timer) * 1e3);
    }

    PRINT_INFO(p.verbosity >= 1, "DPU Kernel Time: %f ms", dpuTime * 1e3);
    PRINT_INFO(p.verbosity >= 1, "Inter-DPU Time: %f ms", hostTime * 1e3);

    // Gather nodeLevel back (uniform padded), then copy only real nodes per DPU
    PRINT_INFO(p.verbosity >= 1, "Copying back the result");
    startTimer(&timer);
    {
        const size_t per_dpu = NODELEVEL_BYTES;
        uint8_t* level_all = (uint8_t*)malloc((size_t)numDPUs * per_dpu);
        void* lvl_ptrs[NR_DPUS];
        for (unsigned d = 0; d < numDPUs; ++d) lvl_ptrs[d] = level_all + (size_t)d * per_dpu;

        gatherFromDPU(&r, dpuLevel_m, (void*(*)[NR_DPUS])&lvl_ptrs, per_dpu);

        for (unsigned d = 0; d < numDPUs; ++d) {
            uint32_t start = d * numNodesPerDPU;
            if (start >= numNodes) continue;
            uint32_t copyN = dpuParams[d].dpuNumNodes;
            memcpy(&nodeLevel[start],
                   level_all + (size_t)d * per_dpu,
                   (size_t)copyN * sizeof(uint32_t));
        }
        free(level_all);
    }
    stopTimer(&timer);
    retrieveTime += getElapsedTime(timer);

    PRINT_INFO(p.verbosity >= 1, "    DPU-CPU Time: %f ms", retrieveTime * 1e3);
    if (p.verbosity == 0) {
        PRINT("CPU-DPU Time(ms): %f    DPU Kernel Time (ms): %f    Inter-DPU Time (ms): %f    DPU-CPU Time (ms): %f",
              loadTime*1e3, dpuTime*1e3, hostTime*1e3, retrieveTime*1e3);
    }

    // CPU reference + verify (kept same as PRIM)
    PRINT_INFO(p.verbosity >= 1, "Calculating result on CPU");
    uint32_t* nodeLevelReference = calloc(numNodes, sizeof(uint32_t));
    memset(nextFrontier, 0, (numNodes/64) * sizeof(uint64_t));
    setBit(nextFrontier[0], 0);
    uint32_t nextEmpty = 0;
    uint32_t lvl = 1;
    while (!nextEmpty) {
        for (uint32_t tile = 0; tile < numTiles; ++tile) {
            uint64_t nf = nextFrontier[tile];
            currentFrontier[tile] = nf;
            if (nf) {
                visited[tile] |= nf;
                nextFrontier[tile] = 0;
                for (uint32_t node = tile*64; node < (tile+1)*64; ++node)
                    if (isSet(nf, node%64)) nodeLevelReference[node] = lvl;
            }
        }
        nextEmpty = 1;
        for (uint32_t tile = 0; tile < numTiles; ++tile) {
            uint64_t cf = currentFrontier[tile];
            if (!cf) continue;
            for (uint32_t node = tile*64; node < (tile+1)*64; ++node) {
                if (!isSet(cf, node%64)) continue;
                uint32_t ptr = nodePtrs[node];
                uint32_t nxt = nodePtrs[node+1];
                for (uint32_t i = ptr; i < nxt; ++i) {
                    uint32_t nb = neighborIdxs[i];
                    if (!isSet(visited[nb/64], nb%64)) {
                        setBit(nextFrontier[nb/64], nb%64);
                        nextEmpty = 0;
                    }
                }
            }
        }
        ++lvl;
    }

    PRINT_INFO(p.verbosity >= 1, "Verifying the result, calclvl:%d", lvl);
    int mism = 0;
    int nodelvl=0, nodelvlref= 0;
    int nodes = 0;
    for (uint32_t n = 0; n < numNodes; ++n) {
        if (nodeLevel[n] != nodeLevelReference[n]) {
            PRINT_ERROR("Mismatch at node %u (CPU=%u, DPU=%u)", n, nodeLevelReference[n], nodeLevel[n]);
            if (++mism > 10) break;
        }
	nodelvl = nodeLevel[n];
	nodelvlref = nodeLevelReference[n];
	nodes = n;
    }
    //printf("nodelvl:%d, nodelvlref:%d nodes:%d\n", nodelvl, nodelvlref, nodes);
        // update CSV
#define TEST_NAME "BFS"
#define RESULTS_FILE "prim_results.csv"
        //update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 0, hostTime, "CPU");
        update_csv(RESULTS_FILE, TEST_NAME, "M_C2D", loadTime*1e3);
        update_csv(RESULTS_FILE, TEST_NAME, "M_D2C", retrieveTime*1e3);
        update_csv(RESULTS_FILE, TEST_NAME, "DPU", dpuTime*1e3);

    freeCOOGraph(cooGraph);
    freeCSRGraph(csrGraph);
    free(nodeLevel);
    free(visited);
    free(currentFrontier);
    free(nextFrontier);
    free(nodeLevelReference);

    vud_rank_free(&r);
    return 0;
}
