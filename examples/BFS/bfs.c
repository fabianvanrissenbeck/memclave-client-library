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

#include "../../src/vud.h"
#include "../../src/vud_mem.h"
#include "../../src/vud_ime.h"
#include "../../src/vud_log.h"
#include "mram-management.h"
#include "support/common.h"
#include "support/graph.h"
#include "support/params.h"
#include "support/timer.h"
#include "support/utils.h"

#define TASKLETS       16
#define NR_DPUS       64
#define ARG_OFFSET    0x3000

#ifndef ENERGY
#define ENERGY 0
#endif
#if ENERGY
#include <dpu_probe.h>
#endif

#define DPU_BINARY "../bfs.sk"

// Main of the Host Application
int main(int argc, char** argv) {

    // Process parameters
    struct Params p = input_params(argc, argv);

    // Timer and profiling
    Timer timer;
    float loadTime = 0.0f, dpuTime = 0.0f, hostTime = 0.0f, retrieveTime = 0.0f;
    #if ENERGY
    struct dpu_probe_t probe;
    DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
    double tenergy=0;
    #endif

    // Allocate DPUs and load binary
    uint32_t numDPUs = NR_DPUS;
    // Allocate and initialize vud rank
    vud_rank r = vud_rank_alloc(1);
    if (r.err) {
        fprintf(stderr, "vud_rank_alloc failed\n");
        return EXIT_FAILURE;
    }
    vud_ime_wait(&r);
    PRINT_INFO(p.verbosity >= 1, "Allocated %d DPU(s)", numDPUs);

    // Initialize BFS data structures
    PRINT_INFO(p.verbosity >= 1, "Reading graph %s", p.fileName);
    struct COOGraph cooGraph = readCOOGraph(p.fileName);
    PRINT_INFO(p.verbosity >= 1, "    Graph has %d nodes and %d edges", cooGraph.numNodes, cooGraph.numEdges);
    struct CSRGraph csrGraph = coo2csr(cooGraph);
    uint32_t numNodes = csrGraph.numNodes;
    uint32_t* nodePtrs = csrGraph.nodePtrs;
    uint32_t* neighborIdxs = csrGraph.neighborIdxs;
    uint32_t* nodeLevel = calloc(numNodes, sizeof(uint32_t)); // Node's BFS level (initially all 0 meaning not reachable)
    uint64_t* visited = calloc(numNodes/64, sizeof(uint64_t)); // Bit vector with one bit per node
    uint64_t* currentFrontier = calloc(numNodes/64, sizeof(uint64_t)); // Bit vector with one bit per node
    uint64_t* nextFrontier = calloc(numNodes/64, sizeof(uint64_t)); // Bit vector with one bit per node
    setBit(nextFrontier[0], 0); // Initialize frontier to first node
    uint32_t level = 1;
    void *rowptr_bufs[NR_DPUS] = {0};
    void *nbr_bufs[NR_DPUS] = {0};
    void *level_bufs[NR_DPUS] = {0};

    // Partition data structure across DPUs
    uint32_t numNodesPerDPU = ROUND_UP_TO_MULTIPLE_OF_64((numNodes - 1)/numDPUs + 1);
    PRINT_INFO(p.verbosity >= 1, "Assigning %u nodes per DPU", numNodesPerDPU);
    struct DPUParams dpuParams[numDPUs];
    unsigned int dpuIdx = 0;
    // ---- Uniform MRAM layout plan (no functional change yet) ----
uint32_t ROWPTR_BYTES   = ROUND_UP_TO_MULTIPLE_OF_8((numNodesPerDPU + 1) * sizeof(uint32_t));

// Find max neighbors among all DPUs' partitions
uint32_t maxNbrs = 0;
for (unsigned d = 0; d < numDPUs; ++d) {
    uint32_t start = d * numNodesPerDPU;
    if (start >= numNodes) break;
    uint32_t end   = start + numNodesPerDPU;
    if (end > numNodes) end = numNodes;
    uint32_t off   = nodePtrs[start];
    uint32_t nbrs  = nodePtrs[end] - off;   // edges in this partition
    if (nbrs > maxNbrs) maxNbrs = nbrs;
}
uint32_t COLIDX_BYTES   = ROUND_UP_TO_MULTIPLE_OF_8(maxNbrs * sizeof(uint32_t));
uint32_t NODELEVEL_BYTES= ROUND_UP_TO_MULTIPLE_OF_8(numNodesPerDPU * sizeof(uint32_t));
uint32_t GLOBAL_BM_BYTES= ROUND_UP_TO_MULTIPLE_OF_8((numNodes/64)     * sizeof(uint64_t));
uint32_t CUR_BM_BYTES   = ROUND_UP_TO_MULTIPLE_OF_8((numNodesPerDPU/64) * sizeof(uint64_t));

// Print once for verification
PRINT_INFO(p.verbosity >= 1, "Uniform sizes (bytes): rowptr=%u, colidx=%u, nodelevel=%u, visited=%u, cur_frontier=%u, next_frontier=%u",
           ROWPTR_BYTES, COLIDX_BYTES, NODELEVEL_BYTES, GLOBAL_BM_BYTES, CUR_BM_BYTES, GLOBAL_BM_BYTES);
    struct mram_heap_allocator_t allocator;
    init_allocator(&allocator);
    for(dpuIdx = 0; dpuIdx < numDPUs; ++dpuIdx) {

        // Allocate parameters
	const uint32_t HEAP_BASE = ARG_OFFSET + ((sizeof(struct DPUParams) + 0xFF) & ~0xFF); // 256B align
	allocator.totalAllocated = HEAP_BASE;

        // Find DPU's nodes
        uint32_t dpuStartNodeIdx = dpuIdx * numNodesPerDPU;
        uint32_t dpuNumNodes;
        if (dpuStartNodeIdx > numNodes) {
            dpuNumNodes = 0;
        } else if (dpuStartNodeIdx + numNodesPerDPU > numNodes) {
            dpuNumNodes = numNodes - dpuStartNodeIdx;
        } else {
            dpuNumNodes = numNodesPerDPU;
        }
        dpuParams[dpuIdx].dpuNumNodes = dpuNumNodes;
        PRINT_INFO(p.verbosity >= 2, "    DPU %u:", dpuIdx);
        PRINT_INFO(p.verbosity >= 2, "        Receives %u nodes", dpuNumNodes);
        // --- build padded rowptr buffer for this DPU (no scatter yet) ---
        {
            uint32_t start = dpuStartNodeIdx;
            uint32_t end   = start + dpuNumNodes;
            if (end > numNodes) end = numNodes;
        
            uint32_t slice_len = (end - start) + 1; // (nodes + 1)
            uint32_t *buf = (uint32_t*)malloc(ROWPTR_BYTES);
            rowptr_bufs[dpuIdx] = buf;
        
            // copy real entries
            memcpy(buf, &nodePtrs[start], slice_len * sizeof(uint32_t));
        
            // pad tail by repeating the last value to keep CSR monotonic
            uint32_t last = buf[slice_len - 1];
            for (uint32_t i = slice_len; i < ROWPTR_BYTES / sizeof(uint32_t); ++i) {
                buf[i] = last;
            }
        }


        // Partition edges and copy data
        if (dpuNumNodes > 0) {

            // Find DPU's CSR graph partition
            uint32_t* dpuNodePtrs_h     = &nodePtrs[dpuStartNodeIdx];
            uint32_t  dpuNodePtrsOffset = dpuNodePtrs_h[0];
            uint32_t* dpuNeighborIdxs_h = neighborIdxs + dpuNodePtrsOffset;
            uint32_t  dpuNumNeighbors   = dpuNodePtrs_h[dpuNumNodes] - dpuNodePtrsOffset;
            uint32_t* dpuNodeLevel_h    = &nodeLevel[dpuStartNodeIdx];
	    {
                uint32_t bytes = dpuNumNeighbors * sizeof(uint32_t);
                uint32_t *buf = (uint32_t*)malloc(COLIDX_BYTES);
                nbr_bufs[dpuIdx] = buf;
   
                memcpy(buf, dpuNeighborIdxs_h, bytes);
                // Pad the remainder with zeros
                memset((uint8_t*)buf + bytes, 0, COLIDX_BYTES - bytes);
	    }
	    {
                uint32_t bytes = dpuNumNodes * sizeof(uint32_t);
                uint32_t *buf = (uint32_t*)malloc(NODELEVEL_BYTES);
                level_bufs[dpuIdx] = buf;
            
                // You probably have nodeLevel[] zero-inited; copy what you do now:
                memcpy(buf, &nodeLevel[dpuStartNodeIdx], bytes);
                // Pad remainder with zeros
                memset((uint8_t*)buf + bytes, 0, NODELEVEL_BYTES - bytes);
            }

            uint32_t dpuNodePtrs_m        = mram_heap_alloc(&allocator, ROWPTR_BYTES);
            uint32_t dpuNeighborIdxs_m    = mram_heap_alloc(&allocator, COLIDX_BYTES);
            uint32_t dpuNodeLevel_m       = mram_heap_alloc(&allocator, NODELEVEL_BYTES);
            uint32_t dpuVisited_m         = mram_heap_alloc(&allocator, GLOBAL_BM_BYTES);
            uint32_t dpuCurrentFrontier_m = mram_heap_alloc(&allocator, CUR_BM_BYTES);
            uint32_t dpuNextFrontier_m    = mram_heap_alloc(&allocator, GLOBAL_BM_BYTES);
	    uint32_t dpuNextFrontierPriv_m = mram_heap_alloc(&allocator, GLOBAL_BM_BYTES * TASKLETS);
            PRINT_INFO(p.verbosity >= 2,
                       "        Total memory allocated is %u bytes",
                       allocator.totalAllocated);

            // Set up DPU parameters
            dpuParams[dpuIdx].numNodes              = numNodes;
            dpuParams[dpuIdx].dpuStartNodeIdx       = dpuStartNodeIdx;
            dpuParams[dpuIdx].dpuNodePtrsOffset     = dpuNodePtrsOffset;
            dpuParams[dpuIdx].level                 = level;
            dpuParams[dpuIdx].dpuNodePtrs_m         = dpuNodePtrs_m;
            dpuParams[dpuIdx].dpuNeighborIdxs_m     = dpuNeighborIdxs_m;
            dpuParams[dpuIdx].dpuNodeLevel_m        = dpuNodeLevel_m;
            dpuParams[dpuIdx].dpuVisited_m          = dpuVisited_m;
            dpuParams[dpuIdx].dpuCurrentFrontier_m  = dpuCurrentFrontier_m;
            dpuParams[dpuIdx].dpuNextFrontier_m     = dpuNextFrontier_m;
	    dpuParams[dpuIdx].dpuNextFrontierPriv_m = dpuNextFrontierPriv_m;
            PRINT_INFO(p.verbosity >= 2,
	    		"dpuNumNodes:%d, numNodes:%d, dpuStartNodeIdx:%d, dpuNodePtrsOffset:%d, level:%d",
	    		dpuNumNodes, numNodes, dpuStartNodeIdx, dpuNodePtrsOffset, level);

            // Send data to DPU
            PRINT_INFO(p.verbosity >= 2, "        Copying data to DPU");
            startTimer(&timer);
            PRINT_INFO(p.verbosity >= 2, "        Broadcast dpuVisited to DPU");
            // --- broadcast the same visited[] and nextFrontier[] to all ---
            copyToDPU(&r,
                      visited,
                      dpuVisited_m,
                      (numNodes/64) * sizeof(uint64_t));
            PRINT_INFO(p.verbosity >= 2, "        Broadcast dpuNextFrontier to DPU");
            copyToDPU(&r,
                      nextFrontier,
                      dpuNextFrontier_m,
                      (numNodes/64) * sizeof(uint64_t));
            // NOTE: No need to copy currentFrontier here, DPU will initialize it
            stopTimer(&timer);
            loadTime += getElapsedTime(timer);
            static uint32_t ref_row=0, ref_col=0, ref_lvl=0, ref_vis=0, ref_cur=0, ref_nxt=0;
            if (dpuIdx == 0) {
                ref_row = dpuNodePtrs_m; ref_col = dpuNeighborIdxs_m; ref_lvl = dpuNodeLevel_m;
                ref_vis = dpuVisited_m;  ref_cur = dpuCurrentFrontier_m; ref_nxt = dpuNextFrontier_m;
            } else {
                assert(dpuNodePtrs_m        == ref_row);
                assert(dpuNeighborIdxs_m    == ref_col);
                assert(dpuNodeLevel_m       == ref_lvl);
                assert(dpuVisited_m         == ref_vis);
                assert(dpuCurrentFrontier_m == ref_cur);
                assert(dpuNextFrontier_m    == ref_nxt);
            }
        }

        // Send parameters to DPU
        PRINT_INFO(p.verbosity >= 2, "        Copying parameters to DPU");
        startTimer(&timer);
        {
            const void* param_slices[NR_DPUS];
            for(int d = 0; d < numDPUs; ++d) {
                param_slices[d] = &dpuParams[d];
            }
            scatterToDPU(&r,
                         (const void*(*)[NR_DPUS])&param_slices,
                         ARG_OFFSET,
                         sizeof(struct DPUParams));
        }
        stopTimer(&timer);
        loadTime += getElapsedTime(timer);
    }
    PRINT_INFO(p.verbosity >= 1, "    CPU-DPU Time: %f ms", loadTime*1e3);
    // ---- single rowptr scatter (all DPUs, same offset/size) ----
    {
        uint32_t ROWPTR_OFFSET = dpuParams[0].dpuNodePtrs_m; // identical across DPUs
        PRINT_INFO(p.verbosity >= 1, "Scatter rowptr (once): %u bytes @ 0x%x",
                   ROWPTR_BYTES, ROWPTR_OFFSET);

        // NOTE the & — we pass a pointer to the array-of-pointers
        scatterToDPU(&r,
                     (const void*(*)[NR_DPUS])&rowptr_bufs,
                     ROWPTR_OFFSET,
                     ROWPTR_BYTES);
        void *chk[NR_DPUS];
        for (unsigned d = 0; d < numDPUs; ++d) chk[d] = malloc(ROWPTR_BYTES);

        gatherFromDPU(&r,
                      dpuParams[0].dpuNodePtrs_m,
                      (void*(*)[NR_DPUS])&chk,     // NOTE the &
                      ROWPTR_BYTES);

        // spot-check
        uint32_t *rp0  = (uint32_t*)chk[0];
        uint32_t *rp63 = (uint32_t*)chk[numDPUs-1];
        PRINT_INFO(1, "Rowptr[DPU0] head=%u, tail=%u",
                   rp0[0],  rp0[(ROWPTR_BYTES/4)-1]);
        PRINT_INFO(1, "Rowptr[DPU%u] head=%u, tail=%u",
                   numDPUs-1,
                   rp63[0], rp63[(ROWPTR_BYTES/4)-1]);

        for (unsigned d = 0; d < numDPUs; ++d) free(chk[d]);
        for (unsigned d = 0; d < numDPUs; ++d) free(rowptr_bufs[d]);
    }
     PRINT_INFO(p.verbosity >= 2, "        Copying neighbor-index slices to DPU");
     {
        uint32_t COLIDX_OFFSET = dpuParams[0].dpuNeighborIdxs_m; // identical across DPUs
        PRINT_INFO(p.verbosity >= 1, "Scatter neighbors (once): %u bytes @ 0x%x",
                   COLIDX_BYTES, COLIDX_OFFSET);
 
        scatterToDPU(&r,
                     (const void*(*)[NR_DPUS])&nbr_bufs,   // NOTE the &
                     COLIDX_OFFSET,
                     COLIDX_BYTES);
 
        for (unsigned d = 0; d < numDPUs; ++d) free(nbr_bufs[d]);
    }
    {
        uint32_t LEVEL_OFFSET = dpuParams[0].dpuNodeLevel_m; // common offset
        PRINT_INFO(p.verbosity >= 1, "Scatter nodeLevel (once): %u bytes @ 0x%x",
                   NODELEVEL_BYTES, LEVEL_OFFSET);
    
        scatterToDPU(&r,
                     (const void*(*)[NR_DPUS])&level_bufs,  // NOTE the &
                     LEVEL_OFFSET,
                     NODELEVEL_BYTES);
    
        for (unsigned d = 0; d < numDPUs; ++d) free(level_bufs[d]);
    }



    // Iterate until next frontier is empty
    uint32_t nextFrontierEmpty = 0;
    uint32_t numTiles = numNodes / 64;
    uint64_t tmpFrontier[numTiles];

    while (!nextFrontierEmpty) {
        PRINT_INFO(p.verbosity >= 1,
                   "Processing current frontier for level %u",
                   level);

    #if ENERGY
        DPU_ASSERT(dpu_probe_start(&probe));
    #endif
        // Run all DPUs
        PRINT_INFO(p.verbosity >= 1, "    Booting DPUs");
        startTimer(&timer);
        vud_ime_launch_sk(&r, DPU_BINARY);
        if (r.err) {
            fprintf(stderr, "vud_ime_launch_sk failed %d\n", r.err);
            return EXIT_FAILURE;
        }
        vud_ime_wait(&r);
        if (r.err) {
            fprintf(stderr, "vud_ime_launch_sk failed %d\n", r.err);
            return EXIT_FAILURE;
        }
        stopTimer(&timer);
        dpuTime += getElapsedTime(timer);
        PRINT_INFO(p.verbosity >= 2,
                   "    Level DPU Time: %f ms",
                   getElapsedTime(timer)*1e3);
    #if ENERGY
        DPU_ASSERT(dpu_probe_stop(&probe));
        double energy;
        DPU_ASSERT(dpu_probe_get(&probe,
                                 DPU_ENERGY,
                                 DPU_AVERAGE,
                                 &energy));
        tenergy += energy;
    #endif

        // Copy back next frontier from all DPUs and compute their union
        startTimer(&timer);
        // reset currentFrontier before merging
        memset(currentFrontier, 0, numTiles*sizeof(uint64_t));
	// 1) allocate one big buffer to hold every DPU’s bitvector
        size_t frontierBytes = numTiles * sizeof(uint64_t);
        uint64_t *allFrontiers = malloc(numDPUs * frontierBytes);
        if(!allFrontiers) {
	    fprintf(stderr, "malloc(allFrontiers) failed\n");
            exit(1);
        }

        // 2) one gather of the entire frontier MRAM region from all DPUs
        copyFromDPU(&r,
            dpuParams[0].dpuNextFrontier_m,  // same MRAM offset in every DPU
            (uint8_t*)allFrontiers,
            frontierBytes);

	// 3) merge each slice into currentFrontier
	for(unsigned d = 0; d < numDPUs; ++d) {
	    uint64_t *thisF = allFrontiers + (size_t)d * numTiles;
	    for(uint32_t t = 0; t < numTiles; ++t) {
	        currentFrontier[t] |= thisF[t];
	    }
	}
#if 1
	uint32_t frontierCount = 0;
	for(uint32_t t = 0; t < numTiles; ++t) {
	    frontierCount += __builtin_popcountll(currentFrontier[t]);
	}
#endif
	free(allFrontiers);

	// 4) test for emptiness
	nextFrontierEmpty = 1;
	for(uint32_t t = 0; t < numTiles; ++t) {
	    if(currentFrontier[t]) {
	        nextFrontierEmpty = 0;
	        break;
	    }
	}

        if (!nextFrontierEmpty) {
            // advance level, then push the new frontier and level back to each DPU
            ++level;
            for (dpuIdx = 0; dpuIdx < numDPUs; ++dpuIdx) {
                if (dpuParams[dpuIdx].dpuNumNodes == 0) {
                    continue;
                }
                // copy the updated frontier in as nextFrontier for the DPU
                copyToDPU(&r,
                          (uint8_t*)currentFrontier,
                          dpuParams[dpuIdx].dpuNextFrontier_m,
                          numTiles * sizeof(uint64_t));
	    }
            // update level
            for (dpuIdx = 0; dpuIdx < numDPUs; ++dpuIdx) {
                dpuParams[dpuIdx].level = level;
	    }
            const void* param_slices[NR_DPUS];
            for (int d = 0; d < numDPUs; ++d) {
                param_slices[d] = &dpuParams[d];
            }
            scatterToDPU(&r,
                         (const void*(*)[NR_DPUS])&param_slices,
                         ARG_OFFSET,
                         sizeof(struct DPUParams));
        }

        stopTimer(&timer);
        hostTime += getElapsedTime(timer);
        PRINT_INFO(p.verbosity >= 2,
                   "    Level Inter-DPU Time: %f ms",
                   getElapsedTime(timer)*1e3);
    }

    PRINT_INFO(p.verbosity >= 1, "DPU Kernel Time: %f ms", dpuTime*1e3);
    PRINT_INFO(p.verbosity >= 1, "Inter-DPU Time: %f ms", hostTime*1e3);
    #if ENERGY
    PRINT_INFO(p.verbosity >= 1, "    DPU Energy: %f J", tenergy);
    #endif

    // Copy back node levels
    PRINT_INFO(p.verbosity >= 1, "Copying back the result");
    startTimer(&timer);
    {
        const uint32_t LEVEL_OFFSET = dpuParams[0].dpuNodeLevel_m; // identical across DPUs
        const uint32_t PER_DPU_BYTES = NODELEVEL_BYTES;            // uniform padded size
        uint8_t *level_all = (uint8_t*)malloc((size_t)numDPUs * PER_DPU_BYTES);
        // Gathers same MRAM region from all DPUs; each DPU lands at +d*PER_DPU_BYTES
        copyFromDPU(&r, LEVEL_OFFSET, level_all, PER_DPU_BYTES);
        for (unsigned d = 0; d < numDPUs; ++d) {
            uint32_t dpuStartNodeIdx = d * numNodesPerDPU;
            uint32_t copyN = dpuParams[d].dpuNumNodes;  // actual nodes on this DPU
            if (copyN) {
                memcpy(&nodeLevel[dpuStartNodeIdx],
                       level_all + (size_t)d * PER_DPU_BYTES,
                       (size_t)copyN * sizeof(uint32_t));
            }
        }
        free(level_all);
    }
    stopTimer(&timer);
    retrieveTime += getElapsedTime(timer);
    PRINT_INFO(p.verbosity >= 1, "    DPU-CPU Time: %f ms", retrieveTime*1e3);
    if(p.verbosity == 0) PRINT("CPU-DPU Time(ms): %f    DPU Kernel Time (ms): %f    Inter-DPU Time (ms): %f    DPU-CPU Time (ms): %f", loadTime*1e3, dpuTime*1e3, hostTime*1e3, retrieveTime*1e3);

    // Calculating result on CPU
    PRINT_INFO(p.verbosity >= 1, "Calculating result on CPU");
    uint32_t* nodeLevelReference = calloc(numNodes, sizeof(uint32_t)); // Node's BFS level (initially all 0 meaning not reachable)
    memset(nextFrontier, 0, numNodes/64*sizeof(uint64_t));
    setBit(nextFrontier[0], 0); // Initialize frontier to first node
    nextFrontierEmpty = 0;
    level = 1;
    while(!nextFrontierEmpty) {
        // Update current frontier and visited list based on the next frontier from the previous iteration
        for(uint32_t nodeTileIdx = 0; nodeTileIdx < numNodes/64; ++nodeTileIdx) {
            uint64_t nextFrontierTile = nextFrontier[nodeTileIdx];
            currentFrontier[nodeTileIdx] = nextFrontierTile;
            if(nextFrontierTile) {
                visited[nodeTileIdx] |= nextFrontierTile;
                nextFrontier[nodeTileIdx] = 0;
                for(uint32_t node = nodeTileIdx*64; node < (nodeTileIdx + 1)*64; ++node) {
                    if(isSet(nextFrontierTile, node%64)) {
                        nodeLevelReference[node] = level;
                    }
                }
            }
        }
        // Visit neighbors of the current frontier
        nextFrontierEmpty = 1;
        for(uint32_t nodeTileIdx = 0; nodeTileIdx < numNodes/64; ++nodeTileIdx) {
            uint64_t currentFrontierTile = currentFrontier[nodeTileIdx];
            if(currentFrontierTile) {
                for(uint32_t node = nodeTileIdx*64; node < (nodeTileIdx + 1)*64; ++node) {
                    if(isSet(currentFrontierTile, node%64)) { // If the node is in the current frontier
                        // Visit its neighbors
                        uint32_t nodePtr = nodePtrs[node];
                        uint32_t nextNodePtr = nodePtrs[node + 1];
                        for(uint32_t i = nodePtr; i < nextNodePtr; ++i) {
                            uint32_t neighbor = neighborIdxs[i];
                            if(!isSet(visited[neighbor/64], neighbor%64)) { // Neighbor not previously visited
                                // Add neighbor to next frontier
                                setBit(nextFrontier[neighbor/64], neighbor%64);
                                nextFrontierEmpty = 0;
                            }
                        }
                    }
                }
            }
        }
        ++level;
    }

    // Verify the result
    PRINT_INFO(p.verbosity >= 1, "Verifying the result");
    //int count = 0;
    for(uint32_t nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
        if(nodeLevel[nodeIdx] != nodeLevelReference[nodeIdx]) {
            PRINT_ERROR("Mismatch at node %u (CPU result = level %u, DPU result = level %u)", nodeIdx, nodeLevelReference[nodeIdx], nodeLevel[nodeIdx]);
	    //count++;
        }
	//if (count > 10) break;
    }

    // Deallocate data structures
    freeCOOGraph(cooGraph);
    freeCSRGraph(csrGraph);
    free(nodeLevel);
    free(visited);
    free(currentFrontier);
    free(nextFrontier);
    free(nodeLevelReference);

    return 0;

}
