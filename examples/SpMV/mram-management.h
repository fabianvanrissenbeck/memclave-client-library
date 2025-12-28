#ifndef _MRAM_MANAGEMENT_H_
#define _MRAM_MANAGEMENT_H_

#include "support/common.h"
#include "support/utils.h"

#include "../../src/vud.h"
#include "../../src/vud_mem.h"

#define DPU_CAPACITY (64 << 20)
#define NR_DPUS      64

struct mram_heap_allocator_t {
    uint32_t totalAllocated;
};

static void init_allocator(struct mram_heap_allocator_t* allocator) {
    allocator->totalAllocated = 0;
}

static uint32_t mram_heap_alloc(struct mram_heap_allocator_t* allocator, uint32_t size) {
    uint32_t ret = allocator->totalAllocated;
    allocator->totalAllocated += ROUND_UP_TO_MULTIPLE_OF_8(size);
    if (allocator->totalAllocated > DPU_CAPACITY) {
        PRINT_ERROR("        Total memory allocated is %u bytes which exceeds the DPU capacity (%u bytes)!",
                    allocator->totalAllocated, DPU_CAPACITY);
        exit(1);
    }
    return ret;
}

// Broadcast same host buffer into MRAM of all DPUs
static void copyToDPU(vud_rank* r, const void* hostPtr, uint32_t mramIdx, uint32_t size) {
    size_t words = (ROUND_UP_TO_MULTIPLE_OF_8(size)) / 8;
    const uint64_t* ptrs[NR_DPUS];
    for (int i = 0; i < NR_DPUS; i++) ptrs[i] = (const uint64_t*)hostPtr;
    vud_simple_transfer(r, words, (const uint64_t(*)[NR_DPUS])&ptrs, mramIdx);
}

// Scatter distinct host slices (src[d]) into each DPU
static void scatterToDPU(vud_rank* r, const void* (*src)[NR_DPUS], uint32_t mramIdx, uint32_t size) {
    size_t words = (ROUND_UP_TO_MULTIPLE_OF_8(size)) / 8;
    vud_simple_transfer(r, words, (const uint64_t(*)[NR_DPUS])src, mramIdx);
}

// Gather same MRAM region from each DPU into per-DPU buffers dst[d]
static void gatherFromDPU(vud_rank* r, uint32_t mramIdx, void* (*dst)[NR_DPUS], uint32_t size) {
    size_t words = (ROUND_UP_TO_MULTIPLE_OF_8(size)) / 8;
    vud_simple_gather(r, words, mramIdx, (uint64_t(*)[NR_DPUS])dst);
}

#endif // _MRAM_MANAGEMENT_H_

