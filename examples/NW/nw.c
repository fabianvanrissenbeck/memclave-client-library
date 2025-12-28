/**
* app.c
* NW Host Application Source File
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <stdint.h>

#include <vud.h>
#include <vud_ime.h>
#include <vud_mem.h>

#define BL 16
#define NR_DPUS     64
#define NR_TASKLETS 16

#include "support/common.h"
#include "support/timer.h"
#include "support/params.h"
#include "support/prim_results.h"

// Subkernel path
#ifndef DPU_BINARY
#define DPU_BINARY "../nw"
#endif

typedef struct {
    uint32_t magic, nblocks, active_blocks, penalty;
} dbg_t;

// Keep consistent with other ports (SpMV)
#define ARG_OFFSET  0x4000u
#define DBG_OFFSET  (ARG_OFFSET + 0x40u)
#define HEAP_BASE   (ARG_OFFSET + 0x100u)   // 256B after args (ALIGN256(sizeof(dpu_arguments_t)))

// ----------------------------------------------------------------------------
// VUD transfer-unit configuration (defaults assume QWORD units)
// ----------------------------------------------------------------------------
#define VUD_XFER_ADDR_IS_BYTES 1
#ifndef VUD_XFER_ADDR_IS_BYTES
#define VUD_XFER_ADDR_IS_BYTES 0
#endif

//#define VUD_XFER_SIZE_IS_BYTES 1
#ifndef VUD_XFER_SIZE_IS_BYTES
#define VUD_XFER_SIZE_IS_BYTES 0
#endif

#define NW_LOGICAL_COLS   (BL + 1) // 17 int32
#define NW_PHYS_COLS      (BL + 2) // 18 int32 (last is padding)

#define NW_ROW_BYTES   ((BL + 2) * (uint32_t)sizeof(int32_t))   // 72
#define NW_ROW_WORDS   (BL + 2)                                // 18
#define NW_LOG_WORDS   (BL + 1)                                // 17

static int32_t *row_pack = NULL;   // NR_DPUS * 18
static int32_t *row_unpack = NULL; // NR_DPUS * 18

static void nw_rowbuf_init(void) {
    if (row_pack && row_unpack) return;

    if (posix_memalign((void **)&row_pack, 8, (size_t)NR_DPUS * NW_ROW_WORDS * sizeof(int32_t)) != 0 || !row_pack) {
        fprintf(stderr, "posix_memalign row_pack failed\n");
        exit(1);
    }
    if (posix_memalign((void **)&row_unpack, 8, (size_t)NR_DPUS * NW_ROW_WORDS * sizeof(int32_t)) != 0 || !row_unpack) {
        fprintf(stderr, "posix_memalign row_unpack failed\n");
        exit(1);
    }
    memset(row_pack, 0, (size_t)NR_DPUS * NW_ROW_WORDS * sizeof(int32_t));
    memset(row_unpack, 0, (size_t)NR_DPUS * NW_ROW_WORDS * sizeof(int32_t));
}

static inline int is_aligned8(const void *p) {
    return (((uintptr_t)p) & 7u) == 0;
}

static inline vud_mram_addr vud_addr_from_byte(uint32_t byte_addr) {
#if VUD_XFER_ADDR_IS_BYTES
    return (vud_mram_addr)byte_addr;
#else
    assert((byte_addr & 7u) == 0);
    return (vud_mram_addr)(byte_addr >> 3);
#endif
}

static inline vud_mram_size vud_size_from_bytes(uint32_t nbytes) {
#if VUD_XFER_SIZE_IS_BYTES
    return (vud_mram_size)nbytes;
#else
    assert((nbytes & 7u) == 0);
    return (vud_mram_size)(nbytes >> 3);
#endif
}

// Helpers
static inline uint32_t umin_u32(uint32_t a, uint32_t b) { return (a < b) ? a : b; }
static inline uint32_t divceil_u32(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// ----------------------------------------------------------------------------
// Robust scatter/gather helpers:
// - Take byte addresses/sizes (like the rest of the code)
// - Convert to whatever the VUD wrapper expects (bytes vs qwords)
// - Bounce through an aligned buffer if any pointer is not 8B aligned
// ----------------------------------------------------------------------------
#define VUD_MAX_XFER_BYTES 128u  // must be >= any single transfer size (we use 64/8/sizeof(args)/sizeof(dbg))

static uint64_t *g_bounce_scatter = NULL;
static uint64_t *g_bounce_gather  = NULL;

static void vud_bounce_init(void) {
    if (g_bounce_scatter && g_bounce_gather) return;

    size_t qwords_per_dpu = (VUD_MAX_XFER_BYTES / 8u);
    size_t total_qwords   = (size_t)NR_DPUS * qwords_per_dpu;

    if (posix_memalign((void **)&g_bounce_scatter, 8, total_qwords * sizeof(uint64_t)) != 0 || !g_bounce_scatter) {
        fprintf(stderr, "posix_memalign g_bounce_scatter failed\n");
        exit(1);
    }
    if (posix_memalign((void **)&g_bounce_gather, 8, total_qwords * sizeof(uint64_t)) != 0 || !g_bounce_gather) {
        fprintf(stderr, "posix_memalign g_bounce_gather failed\n");
        exit(1);
    }
    memset(g_bounce_scatter, 0, total_qwords * sizeof(uint64_t));
    memset(g_bounce_gather,  0, total_qwords * sizeof(uint64_t));
}

static void vud_scatter_bytes(vud_rank *r,
                              const void *src_ptrs[NR_DPUS],
                              uint32_t dst_byte_addr,
                              uint32_t nbytes)
{
    assert((dst_byte_addr & 7u) == 0);
    assert((nbytes & 7u) == 0);
    assert(nbytes <= VUD_MAX_XFER_BYTES);

    vud_bounce_init();

    bool all_aligned = true;
    for (uint32_t i = 0; i < NR_DPUS; i++) {
        if (!is_aligned8(src_ptrs[i])) { all_aligned = false; break; }
    }

    vud_mram_addr dst = vud_addr_from_byte(dst_byte_addr);
    vud_mram_size sz  = vud_size_from_bytes(nbytes);

    const uint64_t *ptrs[NR_DPUS];

    if (all_aligned) {
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            ptrs[i] = (const uint64_t *)src_ptrs[i];
        }
        vud_simple_transfer(r, sz, (const uint64_t * (*)[NR_DPUS])&ptrs, dst);
        return;
    }

    // Bounce: pack each DPU payload into an 8B-aligned slot.
    size_t qwords_per_dpu = (VUD_MAX_XFER_BYTES / 8u);
    for (uint32_t i = 0; i < NR_DPUS; i++) {
        void *slot = (void *)&g_bounce_scatter[(size_t)i * qwords_per_dpu];
        memcpy(slot, src_ptrs[i], nbytes);
        ptrs[i] = (const uint64_t *)slot;
    }

    vud_simple_transfer(r, sz, (const uint64_t * (*)[NR_DPUS])&ptrs, dst);
}

static void vud_gather_bytes(vud_rank *r,
                             uint32_t src_byte_addr,
                             void *dst_ptrs[NR_DPUS],
                             uint32_t nbytes)
{
    assert((src_byte_addr & 7u) == 0);
    assert((nbytes & 7u) == 0);
    assert(nbytes <= VUD_MAX_XFER_BYTES);

    vud_bounce_init();

    bool all_aligned = true;
    for (uint32_t i = 0; i < NR_DPUS; i++) {
        if (!is_aligned8(dst_ptrs[i])) { all_aligned = false; break; }
    }

    vud_mram_addr src = vud_addr_from_byte(src_byte_addr);
    vud_mram_size sz  = vud_size_from_bytes(nbytes);

    uint64_t *ptrs[NR_DPUS];

    if (all_aligned) {
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            ptrs[i] = (uint64_t *)dst_ptrs[i];
        }
        vud_simple_gather(r, sz, src, (uint64_t * (*)[NR_DPUS])&ptrs);
        return;
    }

    // Bounce: gather into aligned slots, then memcpy out.
    size_t qwords_per_dpu = (VUD_MAX_XFER_BYTES / 8u);
    for (uint32_t i = 0; i < NR_DPUS; i++) {
        ptrs[i] = &g_bounce_gather[(size_t)i * qwords_per_dpu];
    }

    vud_simple_gather(r, sz, src, (uint64_t * (*)[NR_DPUS])&ptrs);

    for (uint32_t i = 0; i < NR_DPUS; i++) {
        memcpy(dst_ptrs[i], (const void *)ptrs[i], nbytes);
    }
}

// Traceback in the host
#if PRINT_FILE
static void traceback(int* traceback_output, char *file, int32_t *input_itemsets, int32_t *reference,
                      unsigned int max_rows, unsigned int max_cols, unsigned int penalty) {
    FILE *fpo = fopen(file, "w"); // Use to print to an output file
#else
static void traceback(int* traceback_output, int32_t *input_itemsets, int32_t *reference,
                      unsigned int max_rows, unsigned int max_cols, unsigned int penalty) {
#endif
    int k = 0;
    for (int i = (int)max_rows - 2, j = (int)max_rows - 2; i >= 0 && j >= 0;) {
        int nw = 0, n = 0, w = 0, tb = 0;

#if PRINT_FILE
        if (i == (int)max_rows - 2 && j == (int)max_rows - 2)
            fprintf(fpo, "%d ", input_itemsets[i * max_cols + j]); // print the first element
#endif
        if (i == 0 && j == 0)
            break;

        if (i > 0 && j > 0) {
            nw = input_itemsets[(i - 1) * max_cols + j - 1];
            w  = input_itemsets[i * max_cols + j - 1];
            n  = input_itemsets[(i - 1) * max_cols + j];
        } else if (i == 0) {
            nw = n = LIMIT;
            w  = input_itemsets[i * max_cols + j - 1];
        } else if (j == 0) {
            nw = w = LIMIT;
            n  = input_itemsets[(i - 1) * max_cols + j];
        } else {
            ;
        }

        int new_nw, new_w, new_n;
        new_nw = nw + reference[i * max_cols + j];
        new_w  = w - penalty;
        new_n  = n - penalty;

        tb = maximum(new_nw, new_w, new_n);
        if (tb == new_nw) tb = nw;
        if (tb == new_w)  tb = w;
        if (tb == new_n)  tb = n;

#if PRINT_FILE
        fprintf(fpo, "%d ", tb);
#endif
        traceback_output[k++] = tb;

        if (tb == nw) {
            i--;
            j--;
        } else if (tb == w) {
            j--;
        } else if (tb == n) {
            i--;
        } else {
            ;
        }
    }
    return;
}

// Compute output in the host
static void nw_host(int32_t *input_itemsets, int32_t *reference, uint64_t max_cols, unsigned int penalty) {
    int32_t *input_itemsets_l = (int32_t *)malloc((BL + 1) * (BL + 1) * sizeof(int32_t));
    int32_t *reference_l      = (int32_t *)malloc((BL * BL) * sizeof(int32_t));

    // top-left
    for (uint64_t blk = 1; blk <= (max_cols - 1) / BL; blk++) {
        for (uint64_t b_index_x = 0; b_index_x < blk; b_index_x++) {
            uint64_t b_index_y = blk - 1 - b_index_x;

            for (uint64_t i = 0; i < BL; i++) {
                for (uint64_t j = 0; j < BL; j++) {
                    reference_l[i * BL + j] = reference[(max_cols - 1) * (b_index_y * BL + i) + b_index_x * BL + j];
                }
            }

            for (uint64_t i = 0; i < BL + 1; i++) {
                for (uint64_t j = 0; j < BL + 1; j++) {
                    input_itemsets_l[i * (BL + 1) + j] = input_itemsets[max_cols * (b_index_y * BL + i) + b_index_x * BL + j];
                }
            }

            // Computation
            for (uint64_t i = 1; i < BL + 1; i++) {
                for (uint64_t j = 1; j < BL + 1; j++) {
                    input_itemsets_l[i * (BL + 1) + j] =
                        maximum(input_itemsets_l[(i - 1) * (BL + 1) + j - 1] + reference_l[(i - 1) * BL + j - 1],
                                input_itemsets_l[i * (BL + 1) + j - 1] - penalty,
                                input_itemsets_l[(i - 1) * (BL + 1) + j] - penalty);
                }
            }

            for (uint64_t i = 0; i < BL; i++) {
                for (uint64_t j = 0; j < BL; j++) {
                    input_itemsets[max_cols * (b_index_y * BL + i + 1) + b_index_x * BL + j + 1] =
                        input_itemsets_l[(i + 1) * (BL + 1) + j + 1];
                }
            }
        }
    }

    // bottom-right
    for (uint64_t blk = 2; blk <= (max_cols - 1) / BL; blk++) {
        for (uint64_t b_index_x = blk - 1; b_index_x < (max_cols - 1) / BL; b_index_x++) {
            uint64_t b_index_y = (max_cols - 1) / BL + blk - 2 - b_index_x;

            for (uint64_t i = 0; i < BL; i++) {
                for (uint64_t j = 0; j < BL; j++) {
                    reference_l[i * BL + j] = reference[(max_cols - 1) * (b_index_y * BL + i) + b_index_x * BL + j];
                }
            }

            for (uint64_t i = 0; i < BL + 1; i++) {
                for (uint64_t j = 0; j < BL + 1; j++) {
                    input_itemsets_l[i * (BL + 1) + j] = input_itemsets[max_cols * (b_index_y * BL + i) + b_index_x * BL + j];
                }
            }

            // Computation
            for (uint64_t i = 1; i < BL + 1; i++) {
                for (uint64_t j = 1; j < BL + 1; j++) {
                    input_itemsets_l[i * (BL + 1) + j] =
                        maximum(input_itemsets_l[(i - 1) * (BL + 1) + j - 1] + reference_l[(i - 1) * BL + j - 1],
                                input_itemsets_l[i * (BL + 1) + j - 1] - penalty,
                                input_itemsets_l[(i - 1) * (BL + 1) + j] - penalty);
                }
            }

            for (uint64_t i = 0; i < BL; i++) {
                for (uint64_t j = 0; j < BL; j++) {
                    input_itemsets[max_cols * (b_index_y * BL + i + 1) + b_index_x * BL + j + 1] =
                        input_itemsets_l[(i + 1) * (BL + 1) + j + 1];
                }
            }
        }
    }

    free(input_itemsets_l);
    free(reference_l);
}

int main(int argc, char **argv) {
    struct Params p = input_params(argc, argv);

    // Compile-time layout sanity
    _Static_assert((ARG_OFFSET & 7u) == 0, "ARG_OFFSET must be 8B aligned");
    _Static_assert((DBG_OFFSET & 7u) == 0, "DBG_OFFSET must be 8B aligned");
    _Static_assert((HEAP_BASE  & 7u) == 0, "HEAP_BASE must be 8B aligned");
    _Static_assert((sizeof(dbg_t) % 8) == 0, "dbg_t must be multiple of 8B");
    _Static_assert(sizeof(dpu_arguments_t) <= 0x40, "dpu_arguments_t must fit before DBG_OFFSET");

    // VUD rank + load subkernel
    vud_rank r = vud_rank_alloc(VUD_ALLOC_ANY);
    if (r.err) { fprintf(stderr, "vud_rank_alloc failed\n"); return -1; }

    vud_ime_wait(&r);
    if (r.err) { fprintf(stderr, "vud_ime_wait failed\n"); return -1; }

    vud_ime_load(&r, DPU_BINARY);
    if (r.err) { fprintf(stderr, "cannot load subkernel '%s'\n", DPU_BINARY); return -1; }

    uint32_t nr_of_dpus = NR_DPUS;
    uint32_t max_dpus = nr_of_dpus;

    printf("Allocated %u DPU(s)\n", nr_of_dpus);
    printf("Allocated %u TASKLET(s) per DPU\n", NR_TASKLETS);

    uint64_t max_rows = p.max_rows + 1;
    uint64_t max_cols = p.max_rows + 1;
    unsigned int penalty = p.penalty;

    int32_t *reference            = (int32_t *)malloc(max_rows * max_cols * sizeof(int32_t));
    int32_t *input_itemsets_host  = (int32_t *)malloc(max_rows * max_cols * sizeof(int32_t));
    int32_t *input_itemsets       = (int32_t *)malloc((max_rows + 1) * (max_cols + 1) * sizeof(int32_t));

    // Per-DPU args (aligned for 8B transfers)
    dpu_arguments_t *input_args = NULL;
    if (posix_memalign((void **)&input_args, 8, NR_DPUS * sizeof(dpu_arguments_t)) != 0 || !input_args) {
        fprintf(stderr, "posix_memalign input_args failed\n");
        return -1;
    }
    memset(input_args, 0, NR_DPUS * sizeof(dpu_arguments_t));

    printf("Max size %u\n", p.max_rows);

    // Traceback output
    int32_t *traceback_output      = (int32_t *)malloc((max_rows + max_cols) * sizeof(int32_t));
    int32_t *traceback_output_host = (int32_t *)malloc((max_rows + max_cols) * sizeof(int32_t));
    memset(traceback_output, 0, (max_rows + max_cols) * sizeof(int32_t));
    memset(traceback_output_host, 0, (max_rows + max_cols) * sizeof(int32_t));

    // Dummy buffer (must cover 18 ints for the 72B split transfers)
    int32_t *dummy = NULL;
    if (posix_memalign((void **)&dummy, 8, (BL + 2) * sizeof(int32_t)) != 0 || !dummy) {
        fprintf(stderr, "posix_memalign dummy failed\n");
        return -1;
    }
    memset(dummy, 0, (BL + 2) * sizeof(int32_t));

    // Reusable DBG buffers (avoid leaks)
    dbg_t *dbg = NULL;
    if (posix_memalign((void **)&dbg, 8, NR_DPUS * sizeof(dbg_t)) != 0 || !dbg) {
        fprintf(stderr, "posix_memalign dbg failed\n");
        return -1;
    }
    memset(dbg, 0, NR_DPUS * sizeof(dbg_t));
    void *dbg_ptrs[NR_DPUS];
    for (uint32_t i = 0; i < NR_DPUS; i++) dbg_ptrs[i] = &dbg[i];

    // Timer
    Timer timer;
    Timer long_diagonal_timer;

    for (unsigned int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Initialize itemsets
        for (unsigned int i = 0; i < max_rows; i++) {
            for (unsigned int j = 0; j < max_cols; j++) {
                input_itemsets_host[i * max_cols + j] = 0;
            }
        }

        for (unsigned int i = 0; i <= max_rows; i++) {
            for (unsigned int j = 0; j <= max_cols; j++) {
                input_itemsets[i * (max_cols + 1) + j] = 0;
            }
        }

        // Define random sequences (store in BOTH arrays so reference computation is correct)
        srand(7);
        for (unsigned int i = 1; i < max_rows; i++) {
            int32_t v = (int32_t)(rand() % 10 + 1);
            input_itemsets_host[i * max_cols] = v;
            input_itemsets[i * (max_cols + 1)] = v;
        }

        for (unsigned int j = 1; j < max_cols; j++) {
            int32_t v = (int32_t)(rand() % 10 + 1);
            input_itemsets_host[j] = v;
            input_itemsets[j] = v;
        }

        // Generate reference from the sequences (use input_itemsets with stride max_cols+1)
        for (unsigned int i = 0; i < max_rows - 1; i++) {
            for (unsigned int j = 0; j < max_cols - 1; j++) {
                int32_t a = input_itemsets[(i + 1) * (max_cols + 1) + 0];
                int32_t b = input_itemsets[0 * (max_cols + 1) + (j + 1)];
                reference[i * (max_cols - 1) + j] = blosum62[a][b];
            }
        }

        // Overwrite boundaries with penalties (Rodinia style)
        for (unsigned int i = 1; i < max_rows; i++) {
            input_itemsets_host[i * max_cols] = -((int)i) * (int)penalty;
            input_itemsets[i * (max_cols + 1)] = -((int)i) * (int)penalty;
        }

        for (unsigned int j = 1; j < max_cols; j++) {
            input_itemsets_host[j] = -((int)j) * (int)penalty;
            input_itemsets[j] = -((int)j) * (int)penalty;
        }

        if (rep >= p.n_warmup)
            start(&timer, 0, rep - p.n_warmup);

        // Host CPU computation
        nw_host(input_itemsets_host, reference, max_cols, penalty);

#if PRINT_FILE
        if (rep >= p.n_warmup) {
            char *host_file = "./bin/host_output.txt";
            traceback(traceback_output_host, host_file, input_itemsets_host, reference,
                      (unsigned)max_rows, (unsigned)max_cols, penalty);
        }
#endif
        if (rep >= p.n_warmup)
            stop(&timer, 0);

        // -------------------------
        // Top-left computation on DPUs
        // -------------------------
        for (unsigned int blk = 1; blk <= (max_cols - 1) / BL; blk++) {

#if DYNAMIC
            unsigned int nr_of_blocks = blk;
            uint32_t nr_of_dpus_eff = (nr_of_blocks < max_dpus) ? nr_of_blocks : max_dpus;
#else
            uint32_t nr_of_dpus_eff = nr_of_dpus;
#endif

            // Set up input args (scatter to all 64; inactive DPUs get 0 blocks)
            for (uint32_t i = 0; i < NR_DPUS; i++) {
                input_args[i].nblocks = 0;
                input_args[i].active_blocks = 0;
                input_args[i].penalty = penalty;
                input_args[i].dummy = 0;
            }

            for (uint32_t i = 0; i < nr_of_dpus_eff; i++) {
                unsigned int blocks_per_dpu = blk / nr_of_dpus_eff;
                unsigned int active_blocks_per_dpu = blk / nr_of_dpus_eff;
                unsigned int rest_blocks = blk % nr_of_dpus_eff;
                if (i < rest_blocks)
                    blocks_per_dpu++;
                if (rest_blocks != 0)
                    active_blocks_per_dpu++;

                input_args[i].nblocks = blocks_per_dpu;
                input_args[i].active_blocks = active_blocks_per_dpu; // NOTE: this is the CEIL layout count
                input_args[i].penalty = penalty;
            }

            // Scatter args to MRAM at ARG_OFFSET
            {
                const void *arg_ptrs[NR_DPUS];
                for (uint32_t i = 0; i < NR_DPUS; i++) arg_ptrs[i] = &input_args[i];
                // Ensure size is multiple of 8 for our transfer path
                assert((sizeof(dpu_arguments_t) & 7u) == 0);
                vud_scatter_bytes(&r, arg_ptrs, ARG_OFFSET, (uint32_t)sizeof(dpu_arguments_t));
            }

            // Clear DBG region (avoid stale reads)
            {
                dbg_t z = {0};
                const void *z_ptrs[NR_DPUS];
                for (uint32_t i = 0; i < NR_DPUS; i++) z_ptrs[i] = &z;
                vud_scatter_bytes(&r, z_ptrs, DBG_OFFSET, (uint32_t)sizeof(dbg_t));
            }

            // blocks_per_dpu for transfers (ceil)
            unsigned int blocks_per_dpu = divceil_u32(blk, nr_of_dpus_eff);
            unsigned int mram_offset = 0;

            if (rep >= p.n_warmup) {
                if ((max_cols - 1) / BL == 1)
                    start(&timer, 2, rep - p.n_warmup + blk - 1);
                else
                    start(&timer, 1, rep - p.n_warmup + blk - 1);

                if (blk == ((max_cols - 1) / BL)) {
                    if ((max_cols - 1) / BL == 1)
                        start(&long_diagonal_timer, 2, rep - p.n_warmup);
                    else
                        start(&long_diagonal_timer, 1, rep - p.n_warmup);
                }
            }

            // Copy itemsets to DPUs
            for (unsigned int bl_indx = 0; bl_indx < blocks_per_dpu; bl_indx++) {
                for (unsigned int bl = 0; bl < BL + 1; bl++) {

                    const void *src_ptrs[NR_DPUS];

                    for (uint32_t i = 0; i < NR_DPUS; i++) {
                        if (i >= nr_of_dpus_eff) {
                            src_ptrs[i] = dummy;
                            continue;
                        }

                        unsigned int chunks = blk / nr_of_dpus_eff;
                        unsigned int prev_block_index = 0;
                        unsigned int rest_blocks = blk % nr_of_dpus_eff;

                        if (rest_blocks > 0) {
                            if (i >= rest_blocks) {
                                prev_block_index = rest_blocks * (chunks + 1) + (i - rest_blocks) * chunks;
                            } else {
                                prev_block_index = i * (chunks + 1);
                            }
                        } else {
                            prev_block_index = i * blocks_per_dpu;
                        }

                        uint64_t input_itemsets_offset = 0;
                        int32_t *dpu_pointer = NULL;

                        if (i + bl_indx * nr_of_dpus_eff >= blk) {
                            dpu_pointer = dummy;
                            input_itemsets_offset = 0;
                        } else {
                            uint64_t b_index_x = prev_block_index + bl_indx;
                            uint64_t b_index_y = blk - 1 - b_index_x;
                            dpu_pointer = input_itemsets;
                            input_itemsets_offset =
                                b_index_y * (max_cols + 1) * BL + b_index_x * BL + bl * (max_cols + 1);
                        }

                        src_ptrs[i] = (const void *)(dpu_pointer + input_itemsets_offset);
                    }

                    nw_rowbuf_init();
                   
                    const void *row_ptrs[NR_DPUS];
                   
                    for (uint32_t i = 0; i < NR_DPUS; i++) {
                        int32_t *dst = &row_pack[(size_t)i * NW_ROW_WORDS];
                   
                        if (i >= nr_of_dpus_eff) {
                            memset(dst, 0, NW_ROW_BYTES);
                        } else {
                            const int32_t *src = (const int32_t *)src_ptrs[i]; // points to 17-int logical row
                            memcpy(dst, src, NW_LOG_WORDS * sizeof(int32_t));   // copy 17 ints
                            dst[BL + 1] = 0;                                    // pad int32 (col17)
                        }
                        row_ptrs[i] = dst;
                    }
                   
                    uint32_t dst_mram = HEAP_BASE + mram_offset;
                    vud_scatter_bytes(&r, row_ptrs, dst_mram, NW_ROW_BYTES);
                   
                    mram_offset += NW_ROW_BYTES;
                }
            }

            if (rep >= p.n_warmup) {
                if ((max_cols - 1) / BL == 1)
                    stop(&timer, 2);
                else
                    stop(&timer, 1);

                if (blk == ((max_cols - 1) / BL)) {
                    if ((max_cols - 1) / BL == 1)
                        stop(&long_diagonal_timer, 2);
                    else
                        stop(&long_diagonal_timer, 1);
                }
            }

            // Copy reference to DPUs
            if (rep >= p.n_warmup) {
                start(&timer, 2, rep - p.n_warmup + blk - 1);
                if (blk == ((max_cols - 1) / BL)) {
                    start(&long_diagonal_timer, 2, rep - p.n_warmup);
                }
            }

            mram_offset = blocks_per_dpu * (BL + 1) * (BL + 2) * sizeof(int32_t);

            for (unsigned int bl_indx = 0; bl_indx < blocks_per_dpu; bl_indx++) {
                for (unsigned int bl = 0; bl < BL; bl++) {

                    const void *src_ptrs[NR_DPUS];

                    for (uint32_t i = 0; i < NR_DPUS; i++) {
                        if (i >= nr_of_dpus_eff) {
                            src_ptrs[i] = dummy;
                            continue;
                        }

                        unsigned int chunks = blk / nr_of_dpus_eff;
                        unsigned int prev_block_index = 0;
                        unsigned int rest_blocks = blk % nr_of_dpus_eff;

                        if (rest_blocks > 0) {
                            if (i >= rest_blocks) {
                                prev_block_index = rest_blocks * (chunks + 1) + (i - rest_blocks) * chunks;
                            } else {
                                prev_block_index = i * (chunks + 1);
                            }
                        } else {
                            prev_block_index = i * blocks_per_dpu;
                        }

                        uint64_t reference_offset = 0;
                        int32_t *dpu_pointer = NULL;

                        if (i + bl_indx * nr_of_dpus_eff >= blk) {
                            dpu_pointer = dummy;
                            reference_offset = 0;
                        } else {
                            uint64_t b_index_x = prev_block_index + bl_indx;
                            uint64_t b_index_y = blk - 1 - b_index_x;
                            dpu_pointer = reference;
                            reference_offset =
                                b_index_y * (max_cols - 1) * BL + b_index_x * BL + bl * (max_cols - 1);
                        }

                        src_ptrs[i] = (const void *)(dpu_pointer + reference_offset);
                    }

                    uint32_t dst_mram = HEAP_BASE + mram_offset;
                    // BL * 4 bytes (with BL=16 => 64B)
                    vud_scatter_bytes(&r, src_ptrs, dst_mram, (uint32_t)(BL * sizeof(int32_t)));
                    mram_offset += (unsigned int)(BL * sizeof(int32_t));
                }
            }

            if (rep >= p.n_warmup) {
                stop(&timer, 2);
                if (blk == ((max_cols - 1) / BL)) {
                    stop(&long_diagonal_timer, 2);
                }
            }

            // Launch kernel on DPUs
            if (rep >= p.n_warmup) {
                start(&timer, 3, rep - p.n_warmup + blk - 1);
                if (blk == ((max_cols - 1) / BL)) {
                    start(&long_diagonal_timer, 3, rep - p.n_warmup);
                }
            }

            vud_ime_launch(&r);
            if (r.err) { fprintf(stderr, "vud_ime_launch failed %d\n", r.err); return -1; }
            vud_ime_wait(&r);
            if (r.err) { fprintf(stderr, "vud_ime_wait failed %d\n", r.err); return -1; }

            if (rep >= p.n_warmup) {
                stop(&timer, 3);
                if (blk == ((max_cols - 1) / BL)) {
                    stop(&long_diagonal_timer, 3);
                }
            }

#if 0
            // Read back DPU dbg (optional but useful)
            memset(dbg, 0, NR_DPUS * sizeof(dbg_t));
            vud_gather_bytes(&r, DBG_OFFSET, dbg_ptrs, (uint32_t)sizeof(dbg_t));
            for (int i = 0; i < 2; i++) {
                printf("[DPU dbg] first launch dpu%02d magic=%08x nblocks=%u active=%u pen=%u\n",
                       i, dbg[i].magic, dbg[i].nblocks, dbg[i].active_blocks, dbg[i].penalty);
            }
#endif

            // Retrieve results
            if (rep >= p.n_warmup) {
                start(&timer, 4, rep - p.n_warmup + blk - 1);
                if (blk == ((max_cols - 1) / BL)) {
                    start(&long_diagonal_timer, 4, rep - p.n_warmup);
                }
            }

            mram_offset = 0;
            for (unsigned int bl_indx = 0; bl_indx < blocks_per_dpu; bl_indx++) {
                for (unsigned int bl = 0; bl < BL + 1; bl++) {

                    if (bl == 0) { // skip first row, but advance offset
                        mram_offset += (unsigned int)((BL + 2) * sizeof(int32_t));
                        continue;
                    }

                    void *dst_ptrs0[NR_DPUS];
                    void *dst_ptrs1[NR_DPUS];
                    // Tail gather into per-DPU uint64, then write only col16 back to host matrix.
                    void *tail_dst_ptrs[NR_DPUS];
                    uint64_t tail_words[NR_DPUS];

                    for (uint32_t i = 0; i < NR_DPUS; i++) {

                        if (i >= nr_of_dpus_eff) {
                            dst_ptrs0[i] = dummy;
			    tail_dst_ptrs[i] = &tail_words[i];
                            continue;
                        }

                        unsigned int chunks = blk / nr_of_dpus_eff;
                        unsigned int prev_block_index = 0;
                        unsigned int rest_blocks = blk % nr_of_dpus_eff;

                        if (rest_blocks > 0) {
                            if (i >= rest_blocks) {
                                prev_block_index = rest_blocks * (chunks + 1) + (i - rest_blocks) * chunks;
                            } else {
                                prev_block_index = i * (chunks + 1);
                            }
                        } else {
                            prev_block_index = i * blocks_per_dpu;
                        }

                        uint64_t input_itemsets_offset = 0;
                        int32_t *dpu_pointer = NULL;

                        if (i + bl_indx * nr_of_dpus_eff >= blk) {
                            dpu_pointer = dummy;
                            input_itemsets_offset = 0;
                        } else {
                            uint64_t b_index_x = prev_block_index + bl_indx;
                            uint64_t b_index_y = blk - 1 - b_index_x;
                            dpu_pointer = input_itemsets;
                            input_itemsets_offset =
                                b_index_y * (max_cols + 1) * BL + b_index_x * BL + bl * (max_cols + 1);
                        }

                        int32_t *dst = dpu_pointer + input_itemsets_offset;
                        dst_ptrs0[i] = (void *)dst;         // first 16 ints (64B)
			tail_dst_ptrs[i] = &tail_words[i];
                    }

                    nw_rowbuf_init();
                   
                    if (bl == 0) { // skip row0, but advance offset
                        mram_offset += NW_ROW_BYTES;
                        continue;
                    }
                   
                    void *rowdst[NR_DPUS];
                    for (uint32_t i = 0; i < NR_DPUS; i++) {
                        rowdst[i] = &row_unpack[(size_t)i * NW_ROW_WORDS];
                    }
                   
                    uint32_t src_mram = HEAP_BASE + mram_offset;
                    vud_gather_bytes(&r, src_mram, rowdst, NW_ROW_BYTES);
                   
                    // copy 17 ints back into the host matrix
                    for (uint32_t i = 0; i < NR_DPUS; i++) {
                        if (i >= nr_of_dpus_eff) continue;
                        memcpy(dst_ptrs0[i], rowdst[i], NW_LOG_WORDS * sizeof(int32_t));
                    }
                   
                    mram_offset += NW_ROW_BYTES;

                }
            }

            if (rep >= p.n_warmup) {
                stop(&timer, 4);
                if (blk == ((max_cols - 1) / BL)) {
                    stop(&long_diagonal_timer, 4);
                }
            }
        }

        // -------------------------
        // Bottom-right computation on DPUs
        // -------------------------
        for (unsigned int blk = 2; blk <= (max_cols - 1) / BL; blk++) {

            unsigned int total_blocks = (unsigned int)((max_cols - 1) / BL);
            unsigned int nr_of_blocks = (total_blocks - blk + 1);

#if DYNAMIC
            uint32_t nr_of_dpus_eff = (nr_of_blocks < max_dpus) ? nr_of_blocks : max_dpus;
#else
            uint32_t nr_of_dpus_eff = nr_of_dpus;
#endif

            // Args init
            for (uint32_t i = 0; i < NR_DPUS; i++) {
                input_args[i].nblocks = 0;
                input_args[i].active_blocks = 0;
                input_args[i].penalty = penalty;
                input_args[i].dummy = 0;
            }

            for (uint32_t i = 0; i < nr_of_dpus_eff; i++) {
                unsigned int blocks_per_dpu = nr_of_blocks / nr_of_dpus_eff;
                unsigned int active_blocks_per_dpu = nr_of_blocks / nr_of_dpus_eff;
                unsigned int rest_blocks = nr_of_blocks % nr_of_dpus_eff;

                if (i < rest_blocks)
                    blocks_per_dpu++;
                if (rest_blocks != 0)
                    active_blocks_per_dpu++;

                input_args[i].nblocks = blocks_per_dpu;
                input_args[i].active_blocks = active_blocks_per_dpu; // CEIL layout count
                input_args[i].penalty = penalty;
            }

            // Scatter args
            {
                const void *arg_ptrs[NR_DPUS];
                for (uint32_t i = 0; i < NR_DPUS; i++) arg_ptrs[i] = &input_args[i];
                vud_scatter_bytes(&r, arg_ptrs, ARG_OFFSET, (uint32_t)sizeof(dpu_arguments_t));
            }

            // Clear DBG
            {
                dbg_t z = {0};
                const void *z_ptrs[NR_DPUS];
                for (uint32_t i = 0; i < NR_DPUS; i++) z_ptrs[i] = &z;
                vud_scatter_bytes(&r, z_ptrs, DBG_OFFSET, (uint32_t)sizeof(dbg_t));
            }

            if (rep >= p.n_warmup)
                start(&timer, 1, rep - p.n_warmup + blk - 1);

            unsigned int blocks_per_dpu = divceil_u32(nr_of_blocks, nr_of_dpus_eff);
            unsigned int mram_offset = 0;

            // Copy itemsets to DPUs
            for (unsigned int bl_indx = 0; bl_indx < blocks_per_dpu; bl_indx++) {
                for (unsigned int bl = 0; bl < BL + 1; bl++) {

                    const void *src_ptrs[NR_DPUS];

                    for (uint32_t i = 0; i < NR_DPUS; i++) {

                        if (i >= nr_of_dpus_eff) {
                            src_ptrs[i] = dummy;
                            continue;
                        }

                        unsigned int chunks = nr_of_blocks / nr_of_dpus_eff;
                        unsigned int prev_block_index = 0;
                        unsigned int rest_blocks = nr_of_blocks % nr_of_dpus_eff;

                        if (rest_blocks > 0) {
                            if (i >= rest_blocks) {
                                prev_block_index = rest_blocks * (chunks + 1) + (i - rest_blocks) * chunks;
                            } else {
                                prev_block_index = i * (chunks + 1);
                            }
                        } else {
                            prev_block_index = i * blocks_per_dpu;
                        }

                        uint64_t input_itemsets_offset = 0;
                        int32_t *dpu_pointer = NULL;

                        if (i + bl_indx * nr_of_dpus_eff >= nr_of_blocks) {
                            dpu_pointer = dummy;
                            input_itemsets_offset = 0;
                        } else {
                            uint64_t b_index_x = (uint64_t)(blk - 1) + prev_block_index + bl_indx;
                            uint64_t b_index_y = (max_cols - 1) / BL + blk - 2 - b_index_x;
                            dpu_pointer = input_itemsets;
                            input_itemsets_offset =
                                b_index_y * (max_cols + 1) * BL + b_index_x * BL + bl * (max_cols + 1);
                        }

                        src_ptrs[i] = (const void *)(dpu_pointer + input_itemsets_offset);
                    }

                    uint32_t dst_mram = HEAP_BASE + mram_offset;
                    nw_rowbuf_init();
                   
                    const void *row_ptrs[NR_DPUS];
                   
                    for (uint32_t i = 0; i < NR_DPUS; i++) {
                        int32_t *dst = &row_pack[(size_t)i * NW_ROW_WORDS];
                   
                        if (i >= nr_of_dpus_eff) {
                            memset(dst, 0, NW_ROW_BYTES);
                        } else {
                            const int32_t *src = (const int32_t *)src_ptrs[i]; // points to 17-int logical row
                            memcpy(dst, src, NW_LOG_WORDS * sizeof(int32_t));   // copy 17 ints
                            dst[BL + 1] = 0;                                    // pad int32 (col17)
                        }
                        row_ptrs[i] = dst;
                    }
                   
                    vud_scatter_bytes(&r, row_ptrs, dst_mram, NW_ROW_BYTES);
                   
                    mram_offset += NW_ROW_BYTES;
                }
            }

            if (rep >= p.n_warmup)
                stop(&timer, 1);

            if (rep >= p.n_warmup)
                start(&timer, 2, rep - p.n_warmup + blk - 1);

            // Copy reference to DPUs
            mram_offset = blocks_per_dpu * (BL + 1) * (BL + 2) * sizeof(int32_t);

            for (unsigned int bl_indx = 0; bl_indx < blocks_per_dpu; bl_indx++) {
                for (unsigned int bl = 0; bl < BL; bl++) {

                    const void *src_ptrs[NR_DPUS];

                    for (uint32_t i = 0; i < NR_DPUS; i++) {

                        if (i >= nr_of_dpus_eff) {
                            src_ptrs[i] = dummy;
                            continue;
                        }

                        unsigned int chunks = nr_of_blocks / nr_of_dpus_eff;
                        unsigned int prev_block_index = 0;
                        unsigned int rest_blocks = nr_of_blocks % nr_of_dpus_eff;

                        if (rest_blocks > 0) {
                            if (i >= rest_blocks) {
                                prev_block_index = rest_blocks * (chunks + 1) + (i - rest_blocks) * chunks;
                            } else {
                                prev_block_index = i * (chunks + 1);
                            }
                        } else {
                            prev_block_index = i * blocks_per_dpu;
                        }

                        uint64_t reference_offset = 0;
                        int32_t *dpu_pointer = NULL;

                        if (i + bl_indx * nr_of_dpus_eff >= nr_of_blocks) {
                            dpu_pointer = dummy;
                            reference_offset = 0;
                        } else {
                            uint64_t b_index_x = (uint64_t)(blk - 1) + prev_block_index + bl_indx;
                            uint64_t b_index_y = (max_cols - 1) / BL + blk - 2 - b_index_x;
                            dpu_pointer = reference;
                            reference_offset =
                                b_index_y * (max_cols - 1) * BL + b_index_x * BL + bl * (max_cols - 1);
                        }

                        src_ptrs[i] = (const void *)(dpu_pointer + reference_offset);
                    }

                    uint32_t dst_mram = HEAP_BASE + mram_offset;
                    vud_scatter_bytes(&r, src_ptrs, dst_mram, (uint32_t)(BL * sizeof(int32_t)));
                    mram_offset += (unsigned int)(BL * sizeof(int32_t));
                }
            }

            if (rep >= p.n_warmup)
                stop(&timer, 2);

            // Launch kernel
            if (rep >= p.n_warmup)
                start(&timer, 3, rep - p.n_warmup + blk - 1);

            vud_ime_launch(&r);
            if (r.err) { fprintf(stderr, "vud_ime_launch failed %d\n", r.err); return -1; }
            vud_ime_wait(&r);
            if (r.err) { fprintf(stderr, "vud_ime_wait failed %d\n", r.err); return -1; }

            if (rep >= p.n_warmup)
                stop(&timer, 3);

#if 0
            memset(dbg, 0, NR_DPUS * sizeof(dbg_t));
            vud_gather_bytes(&r, DBG_OFFSET, dbg_ptrs, (uint32_t)sizeof(dbg_t));
            for (int i = 0; i < 2; i++) {
                printf("[DPU dbg] second launch dpu%02d magic=%08x nblocks=%u active=%u pen=%u\n",
                       i, dbg[i].magic, dbg[i].nblocks, dbg[i].active_blocks, dbg[i].penalty);
            }
#endif

            // Retrieve results
            if (rep >= p.n_warmup)
                start(&timer, 4, rep - p.n_warmup + blk - 1);

            mram_offset = 0;
            for (unsigned int bl_indx = 0; bl_indx < blocks_per_dpu; bl_indx++) {
                for (unsigned int bl = 0; bl < BL + 1; bl++) {

                    if (bl == 0) {
                        mram_offset += (unsigned int)((BL + 2) * sizeof(int32_t));
                        continue;
                    }

		    void *dst_ptrs0[NR_DPUS];
                    void *tail_dst_ptrs[NR_DPUS];
                    uint64_t tail_words[NR_DPUS];

                    for (uint32_t i = 0; i < NR_DPUS; i++) {

                        if (i >= nr_of_dpus_eff) {
                            dst_ptrs0[i] = dummy;
			    tail_dst_ptrs[i] = &tail_words[i];
                            continue;
                        }

                        unsigned int chunks = nr_of_blocks / nr_of_dpus_eff;
                        unsigned int prev_block_index = 0;
                        unsigned int rest_blocks = nr_of_blocks % nr_of_dpus_eff;

                        if (rest_blocks > 0) {
                            if (i >= rest_blocks) {
                                prev_block_index = rest_blocks * (chunks + 1) + (i - rest_blocks) * chunks;
                            } else {
                                prev_block_index = i * (chunks + 1);
                            }
                        } else {
                            prev_block_index = i * blocks_per_dpu;
                        }

                        uint64_t input_itemsets_offset = 0;
                        int32_t *dpu_pointer = NULL;

                        if (i + bl_indx * nr_of_dpus_eff >= nr_of_blocks) {
                            dpu_pointer = dummy;
                            input_itemsets_offset = 0;
                        } else {
                            uint64_t b_index_x = (uint64_t)(blk - 1) + prev_block_index + bl_indx;
                            uint64_t b_index_y = (max_cols - 1) / BL + blk - 2 - b_index_x;
                            dpu_pointer = input_itemsets;
                            input_itemsets_offset =
                                b_index_y * (max_cols + 1) * BL + b_index_x * BL + bl * (max_cols + 1);
                        }

                        int32_t *dst = dpu_pointer + input_itemsets_offset;
                        dst_ptrs0[i] = (void *)dst;
                        tail_dst_ptrs[i] = &tail_words[i];
                    }

                    nw_rowbuf_init();
                   
                    if (bl == 0) { // skip row0, but advance offset
                        mram_offset += NW_ROW_BYTES;
                        continue;
                    }
                   
                    void *rowdst[NR_DPUS];
                    for (uint32_t i = 0; i < NR_DPUS; i++) {
                        rowdst[i] = &row_unpack[(size_t)i * NW_ROW_WORDS];
                    }
                   
                    uint32_t src_mram = HEAP_BASE + mram_offset;
                    vud_gather_bytes(&r, src_mram, rowdst, NW_ROW_BYTES);
                   
                    // copy 17 ints back into the host matrix
                    for (uint32_t i = 0; i < NR_DPUS; i++) {
                        if (i >= nr_of_dpus_eff) continue;
                        memcpy(dst_ptrs0[i], rowdst[i], NW_LOG_WORDS * sizeof(int32_t));
                    }
                   
                    mram_offset += NW_ROW_BYTES;
                }
            }

            if (rep >= p.n_warmup)
                stop(&timer, 4);
        }

        // Traceback step (inter-DPU stage in PRIM)
        if (rep >= p.n_warmup)
            start(&timer, 1, 1);

#if PRINT_FILE
        char *dpu_file = "./bin/dpu_output.txt";
        traceback(traceback_output, dpu_file, input_itemsets, reference,
                  (unsigned)(max_rows + 1), (unsigned)(max_cols + 1), penalty);
#else
        traceback(traceback_output, input_itemsets, reference,
                  (unsigned)(max_rows + 1), (unsigned)(max_cols + 1), penalty);
#endif

        if (rep >= p.n_warmup)
            stop(&timer, 1);
    }

    // Print timing results
    printf("CPU version ");
    print(&timer, 0, p.n_reps);
    printf("CPU-DPU ");
    print(&timer, 2, p.n_reps);
    printf("DPU Kernel ");
    print(&timer, 3, p.n_reps);
    printf("Inter-DPU ");
    print(&timer, 1, p.n_reps);
    printf("DPU-CPU ");
    print(&timer, 4, p.n_reps);
    printf("\n");
    printf("Longest Diagonal CPU-DPU ");
    print(&long_diagonal_timer, 2, p.n_reps);
    printf("Longest Diagonal DPU Kernel ");
    print(&long_diagonal_timer, 3, p.n_reps);
    printf("Longest Diagonal Inter-DPU ");
    print(&long_diagonal_timer, 1, p.n_reps);
    printf("Longest Diagonal DPU-CPU ");
    print(&long_diagonal_timer, 4, p.n_reps);
    printf("\n");

    // update CSV
#define TEST_NAME "NW"
#define RESULTS_FILE "prim_results.csv"
    //update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 0, p.n_reps, "CPU");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 2, p.n_reps, "M_C2D");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 4, p.n_reps, "M_D2C");
    update_csv_from_timer(RESULTS_FILE, TEST_NAME, &timer, 3, p.n_reps, "DPU");
    //double dpu_ms = prim_timer_ms_avg(&timer, 2, p.n_reps) + prim_timer_ms_avg(&timer, 3, p.n_reps);
    //update_csv(RESULTS_FILE, TEST_NAME, "DPU", dpu_ms);

    // Check output
    bool status = true;
    for (uint64_t i = 1; i < max_rows; i++) {
        for (uint64_t j = 1; j < max_cols; j++) {
            if (input_itemsets_host[i * max_cols + j] != input_itemsets[i * (max_cols + 1) + j]) {
                status = false;
#if PRINT
                printf("%lu (%lu, %lu): %d %d\n",
                       i * max_cols + j, i, j,
                       input_itemsets_host[i * max_cols + j],
                       input_itemsets[i * (max_cols + 1) + j]);
#endif
            }
        }
    }

    if (status) {
        printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }

    free(input_itemsets_host);
    free(input_itemsets);
    free(reference);
    free(traceback_output);
    free(traceback_output_host);
    free(input_args);
    free(dummy);
    free(dbg);

    free(g_bounce_scatter);
    free(g_bounce_gather);
    free(row_pack);
    free(row_unpack);


    vud_rank_free(&r);
    return status ? 0 : -1;
}
