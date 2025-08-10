#ifndef _MRAM_MANAGEMENT_H_
#define _MRAM_MANAGEMENT_H_

#include "support/common.h"
#include "support/utils.h"

// bring in vud APIs
#include "../../src/vud.h"
#include "../../src/vud_mem.h"

#define DPU_CAPACITY (64 << 20) // A DPU's capacity is 64 MiB
#define NR_DPUS      64         // must match your host’s NR_DPUS

struct mram_heap_allocator_t {
    uint32_t totalAllocated;
};

static void init_allocator(struct mram_heap_allocator_t* allocator) {
    allocator->totalAllocated = 0;
}

static uint32_t mram_heap_alloc(struct mram_heap_allocator_t* allocator,
                                uint32_t size)
{
    uint32_t ret = allocator->totalAllocated;
    allocator->totalAllocated += ROUND_UP_TO_MULTIPLE_OF_8(size);
    if (allocator->totalAllocated > DPU_CAPACITY) {
        PRINT_ERROR("        Total memory allocated is %u bytes which exceeds "
                    "the DPU capacity (%u bytes)!",
                    allocator->totalAllocated, DPU_CAPACITY);
        exit(1);
    }
    return ret;
}

/**
 * Copy a single host buffer (same data) into the MRAM of all DPUs.
 * r         – pointer to your vud_rank
 * hostPtr   – host pointer to copy from
 * mramIdx   – MRAM byte offset in each DPU
 * size      – number of bytes to copy
 */
static void copyToDPU(vud_rank* r,
                      const void* hostPtr,
                      uint32_t    mramIdx,
                      uint32_t    size)
{
    // round up to words
    size_t words = (ROUND_UP_TO_MULTIPLE_OF_8(size)) / 8;
    // build an array of identical pointers
    const uint64_t* ptrs[NR_DPUS];
    for (int i = 0; i < NR_DPUS; i++) {
        ptrs[i] = (const uint64_t*)hostPtr;
    }
    // scatter/broadcast
    vud_simple_transfer(r,
                        words,
                        (const uint64_t(*)[NR_DPUS])&ptrs,
                        mramIdx);
}

/**
 * Gather the same MRAM region from each DPU back into a single host
 * buffer.  This will overwrite hostPtr repeatedly (last DPU’s slice
 * lands at hostPtr + rank*words), so you probably only want this when
 * each DPU is writing to a distinct slice of hostPtr.
 *
 * r         – pointer to your vud_rank
 * mramIdx   – MRAM byte offset in each DPU
 * hostPtr   – host buffer (must be large enough for NR_DPUS * size)
 * size      – number of bytes *per* DPU
 */
static void copyFromDPU(vud_rank* r,
                        uint32_t    mramIdx,
                        void*       hostPtr,
                        uint32_t    size)
{
    size_t words = (ROUND_UP_TO_MULTIPLE_OF_8(size)) / 8;
    uint64_t* dsts[NR_DPUS];
    for (int i = 0; i < NR_DPUS; i++) {
        // each DPU writes into its own slice of hostPtr
        dsts[i] = (uint64_t*)( (char*)hostPtr + i * size );
    }
    vud_simple_gather(r,
                      words,
                      mramIdx,
                      (uint64_t(*)[NR_DPUS])&dsts);
}

/// Scatter *distinct* host_slices[d] (each of `size` bytes) into
/// DPU-d’s MRAM at offset mramIdx
static void scatterToDPU(vud_rank*       r,
                         const void*     (*src)[NR_DPUS],
                         uint32_t        mramIdx,
                         uint32_t        size)
{
    size_t words = (ROUND_UP_TO_MULTIPLE_OF_8(size)) / 8;
    vud_simple_transfer(r,
                        words,
                        (const uint64_t(*)[NR_DPUS])src,
                        mramIdx);
}

/// Gather from all DPUs into per-DPU buffers:
/// dst[d] must point to a buffer with at least `size` bytes.
static void gatherFromDPU(
    vud_rank*    r,
    uint32_t     mramIdx,
    void*        (*dst)[NR_DPUS],   // pointer to array-of-NR_DPUS pointers
    uint32_t     size)
{
    size_t words = (ROUND_UP_TO_MULTIPLE_OF_8(size)) / 8;
    // vud_simple_gather expects a pointer-to-array of uint64_t*
    vud_simple_gather(r, words, mramIdx, (uint64_t(*)[NR_DPUS])dst);
}
#endif // _MRAM_MANAGEMENT_H_
