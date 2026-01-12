#include "vud_mem.h"
#include "vud_sk.h"

#include <stdio.h>
#include <assert.h>

typedef enum mem_op_type {
    MEM_OP_BROADCAST = 1,
    MEM_OP_TRANSFER,
    MEM_OP_GATHER
} mem_op_type;

typedef struct mem_op {
    vud_rank* rank;
    mem_op_type type;
    union {
        struct {
            vud_mram_size sz;
            const uint64_t (*src)[];
            vud_mram_addr tgt;
        } bc;
        struct {
            vud_mram_size sz;
            const uint64_t* (*src)[64];
            vud_mram_addr tgt;
        } tf;
        struct {
            vud_mram_size sz;
            vud_mram_addr src;
            uint64_t* (*tgt)[64];
        } gt;
    };
} mem_op;

static vud_mram_addr virt_to_real(vud_mram_addr addr) {
    const vud_mram_addr mask_0_13 = 0x3FFF;
    const vud_mram_addr mask_14 = 0x4000;
    const vud_mram_addr mask_15_21 = 0x3F8000;
    const vud_mram_addr mask_22_25 = 0x3C00000;

    return (addr & mask_0_13) | ((addr & mask_15_21) >> 1) | (addr & mask_14) << 7 | (addr & mask_22_25);
}

static void byte_interleave_mat(uint64_t (*mat)[8]) {
    uint8_t (*bytes)[8][8] = (void*) mat;

    for (int i = 0; i < 8; ++i) {
        for (int j = i + 1; j < 8; ++j) {
            (*bytes)[i][j] ^= (*bytes)[j][i]; // a = a ^ b
            (*bytes)[j][i] ^= (*bytes)[i][j]; // b = b ^ a ^ b = a
            (*bytes)[i][j] ^= (*bytes)[j][i]; // a = a ^ b ^ a = b
        }
    }
}

static void mat_to_mem(const uint64_t (*mat)[8], volatile uint64_t* mem) {
    __asm__ volatile(
        "movq 0(%0), %%mm0\n"
        "movntq %%mm0, 0(%1)\n"
        "movq 8(%0), %%mm0\n"
        "movntq %%mm0, 8(%1)\n"
        "movq 16(%0), %%mm0\n"
        "movntq %%mm0, 16(%1)\n"
        "movq 24(%0), %%mm0\n"
        "movntq %%mm0, 24(%1)\n"
        "movq 32(%0), %%mm0\n"
        "movntq %%mm0, 32(%1)\n"
        "movq 40(%0), %%mm0\n"
        "movntq %%mm0, 40(%1)\n"
        "movq 48(%0), %%mm0\n"
        "movntq %%mm0, 48(%1)\n"
        "movq 56(%0), %%mm0\n"
        "movntq %%mm0, 56(%1)\n"
	"emms\n"
        :
        : "r" (&(*mat)[0]), "r" (mem)
        : "mm0", "memory"
    );
}

static void mem_to_mat(const volatile uint64_t* mem, uint64_t (*mat)[8]) {
    (*mat)[0] = mem[0];
    (*mat)[1] = mem[1];
    (*mat)[2] = mem[2];
    (*mat)[3] = mem[3];
    (*mat)[4] = mem[4];
    (*mat)[5] = mem[5];
    (*mat)[6] = mem[6];
    (*mat)[7] = mem[7];
}

static void flush_cache_line(volatile uint64_t* mem) {
    __asm__ volatile(
        "clflushopt 0(%0)\n"
        :
        : "r" (mem) : "memory"
    );
}

static void invoc_memory_fence(void) {
    __asm__ volatile("mfence\n"::: "memory");
}

static volatile uint64_t* line_for_group(vud_rank* r, vud_mram_addr addr, unsigned group_nr) {
    vud_mram_addr real = virt_to_real(addr);
    uintptr_t chunk_nr = real / 0x2000;
    uintptr_t block_nr = (real % 0x2000) / 8;
    uintptr_t off = 0x40000 * (group_nr % 4) + 0x40 * (group_nr >= 4) + chunk_nr * 0x100000 + block_nr * 0x80;

    return (volatile uint64_t*)((uintptr_t) r->base + off);
}

static unsigned get_dpu_id(unsigned group_nr, unsigned ci_nr) {
    return ci_nr * 8 + group_nr;
}

static void intl_broadcast_transfer(vud_rank* r, vud_mram_size sz, const uint64_t (*src)[sz], vud_mram_addr tgt, unsigned id, unsigned nr_worker) {
    for (size_t i = id; i < sz; i += nr_worker) {
        uint64_t w = (*src)[i];
        uint64_t mat[8] = { w, w, w, w, w, w, w, w };

        byte_interleave_mat(&mat);

        vud_mram_addr addr = tgt + i * 8;

        // copy for group pairs directly to do less address calculation
        for (int group_nr = 0; group_nr < 4; ++group_nr) {
            volatile uint64_t* line = line_for_group(r, addr, group_nr);

            mat_to_mem(&mat, line);
            mat_to_mem(&mat, &line[8]);
        }
    }

    invoc_memory_fence();
}

static void intl_simple_transfer(vud_rank* r, vud_mram_size sz, const uint64_t* (*src)[64], vud_mram_addr tgt, unsigned id, unsigned nr_worker) {
    unsigned n_unaligned = (1024 - (tgt / 8) % 1024) % 1024;

    for (unsigned group_nr = id; group_nr < 8; group_nr += nr_worker) {
        for (size_t i = 0; i < n_unaligned && i < sz; ++i) {
            uint64_t mat[8];

            for (size_t j = 0; j < 8; ++j) {
                mat[j] = (*src)[j * 8 + group_nr][i];
            }

            vud_mram_addr addr = tgt + i * 8;
            volatile uint64_t* line = line_for_group(r, addr, group_nr);

            byte_interleave_mat(&mat);
            mat_to_mem(&mat, line);
        }
    }

    for (size_t k = id * 1024; k < sz; k += nr_worker * 1024) {
        for (unsigned group_nr = 0; group_nr < 8; group_nr++) {
            for (size_t i = 0; i < 1024 && i + k + n_unaligned < sz; ++i) {
                uint64_t mat[8];

                for (size_t j = 0; j < 8; ++j) {
                    mat[j] = (*src)[j * 8 + group_nr][i + k + n_unaligned];
                }

                vud_mram_addr addr = tgt + (i + k + n_unaligned) * 8;
                volatile uint64_t* line = line_for_group(r, addr, group_nr);

                byte_interleave_mat(&mat);
                mat_to_mem(&mat, line);
            }
        }
    }

    invoc_memory_fence();
}

static void intl_simple_gather(vud_rank* r, vud_mram_size sz, vud_mram_addr src, uint64_t* (*tgt)[64], unsigned id, unsigned nr_worker) {
    // flush all relevant cache lines

    invoc_memory_fence();

    for (size_t i = 0; i < sz; i += 1) {
        vud_mram_addr addr = src + i * 8;

        for (unsigned group_nr = 0; group_nr < 8; ++group_nr) {
            volatile uint64_t* line = line_for_group(r, addr, group_nr);
            flush_cache_line(line);
        }
    }

    invoc_memory_fence();

    // copy the first words so that src becomes 1024 word aligned
    // this makes the following copy slightly more efficient

    unsigned n_unaligned = (1024 - (src / 8) % 1024) % 1024;

    for (unsigned group_nr = id; group_nr < 8; group_nr += nr_worker) {
        for (size_t i = 0; i < sz && i < n_unaligned; ++i) {
            vud_mram_addr addr = src + i * 8;
            volatile uint64_t* line = line_for_group(r, addr, group_nr);

            uint64_t mat[8];

            mem_to_mat(line, &mat);
            byte_interleave_mat(&mat);

            for (unsigned ci_nr = 0; ci_nr < 8; ++ci_nr) {
                (*tgt)[get_dpu_id(group_nr, ci_nr)][i] = mat[ci_nr];
            }
        }
    }

    // UPMEM arranges memory in a way that causes 1024 words of one DPU to be
    // in one contiguous region of memory (still transposed and everything)
    // By doing 1024 * nr_worker word steps we can distribute work without
    // causing inefficiencies due to non-local reads

    for (size_t j = id * 1024; j < sz; j += 1024 * nr_worker) {
        for (unsigned group_nr = 0; group_nr < 8; ++group_nr) {
            for (size_t i = 0; i + j + n_unaligned < sz && i < 1024; ++i) {
                assert((src + (j + n_unaligned) * 8) % 8192 == 0);

                vud_mram_addr addr = src + (i + j + n_unaligned) * 8;
                volatile uint64_t* line = line_for_group(r, addr, group_nr);

                uint64_t mat[8];

                mem_to_mat(line, &mat);
                byte_interleave_mat(&mat);

                for (unsigned ci_nr = 0; ci_nr < 8; ++ci_nr) {
                    (*tgt)[get_dpu_id(group_nr, ci_nr)][i + j + n_unaligned] = mat[ci_nr];
                }
            }
        }
    }

    invoc_memory_fence();

    for (size_t i = 0; i < sz; i += 1) {
        vud_mram_addr addr = src + i * 8;

        for (unsigned group_nr = 0; group_nr < 8; ++group_nr) {
            volatile uint64_t* line = line_for_group(r, addr, group_nr);
            flush_cache_line(line);
        }
    }

    invoc_memory_fence();
}

static void pool_op_worker(unsigned id, unsigned nr_worker, void* arg_ptr) {
    mem_op* arg = arg_ptr;

    switch (arg->type) {
    case MEM_OP_BROADCAST:
        intl_broadcast_transfer(arg->rank, arg->bc.sz, arg->bc.src, arg->bc.tgt, id, nr_worker);
        break;

    case MEM_OP_TRANSFER:
        intl_simple_transfer(arg->rank, arg->tf.sz, arg->tf.src, arg->tf.tgt, id, nr_worker);
        break;

    case MEM_OP_GATHER:
        intl_simple_gather(arg->rank, arg->gt.sz, arg->gt.src, arg->gt.tgt, id, nr_worker);
        break;
    }
}

static void pool_do_op(vud_rank* r, mem_op op) {
    op.rank = r;
    vud_pool_do(r->pool, pool_op_worker, &op);
}

void vud_broadcast_transfer(vud_rank* r, vud_mram_size sz, const uint64_t (*src)[sz], vud_mram_addr tgt) {
    pool_do_op(r, (mem_op) {
        .type = MEM_OP_BROADCAST,
        .bc = {
            .sz = sz,
            .src = src,
            .tgt = tgt
        }
    });
}

void vud_simple_transfer(vud_rank* r, vud_mram_size sz, const uint64_t* (*src)[64], vud_mram_addr tgt) {
    pool_do_op(r, (mem_op) {
        .type = MEM_OP_TRANSFER,
        .tf = {
            .sz = sz,
            .src = src,
            .tgt = tgt
        }
    });
}

void vud_simple_gather(vud_rank* r, vud_mram_size sz, vud_mram_addr src, uint64_t* (*tgt)[64]) {
    pool_do_op(r, (mem_op) {
        .type = MEM_OP_GATHER,
        .gt = {
            .sz = sz,
            .src = src,
            .tgt = tgt
        }
    });
}

void vud_broadcast_to(vud_rank* r, vud_mram_size sz, const uint64_t (*src)[sz], const char* symbol) {
    vud_mram_addr addr;
    vud_error err;

    if (r->err == VUD_OK) {
        if ((err = vud_get_symbol(r->next_sk, symbol, &addr))) {
            r->err = err;
        }
    }

    vud_broadcast_transfer(r, sz, src, addr);
}

void vud_transfer_to(vud_rank* r, vud_mram_size sz, const uint64_t* (*src)[64], const char* symbol) {
    vud_mram_addr addr;
    vud_error err;

    if (r->err == VUD_OK) {
        if ((err = vud_get_symbol(r->next_sk, symbol, &addr))) {
            r->err = err;
        }
    }

    vud_simple_transfer(r, sz, src, addr);
}

void vud_gather_from(vud_rank* r, vud_mram_size sz, const char* symbol, uint64_t* (*tgt)[64]) {
    vud_mram_addr addr;
    vud_error err;

    if (r->err == VUD_OK) {
        if ((err = vud_get_symbol(r->next_sk, symbol, &addr))) {
            r->err = err;
        }
    }

    vud_simple_gather(r, sz, addr, tgt);
}
