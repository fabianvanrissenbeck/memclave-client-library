#include "vud_mem.h"

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
    __asm__(
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
        :
        : "r" (&(*mat)[0]), "r" (mem)
        : "mm0"
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
    __asm__(
        "clflushopt 0(%0)\n"
        :
        : "r" (mem)
    );
}

static void invoc_memory_fence(void) {
    __asm__("mfence\n");
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

void vud_broadcast_transfer(vud_rank* r, vud_mram_size sz, const uint64_t (*src)[sz], vud_mram_addr tgt) {
    for (size_t i = 0; i < sz; ++i) {
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

void vud_simple_transfer(vud_rank* r, vud_mram_size sz, const uint64_t* (*src)[64], vud_mram_addr tgt) {
    for (size_t i = 0; i < sz; ++i) {
        vud_mram_addr addr = tgt + i * 8;

        for (unsigned group_nr = 0; group_nr < 8; ++group_nr) {
            volatile uint64_t* line = line_for_group(r, addr, group_nr);
            uint64_t mat[8];

            for (size_t j = 0; j < 8; ++j) {
                mat[j] = (*src)[j * 8 + group_nr][i];
            }

            byte_interleave_mat(&mat);
            mat_to_mem(&mat, line);
        }
    }

    invoc_memory_fence();
}

void vud_simple_gather(vud_rank* r, vud_mram_size sz, vud_mram_addr src, uint64_t* (*tgt)[64]) {
    // flush all relevant cache lines

    invoc_memory_fence();

    for (size_t i = 0; i < sz; ++i) {
        vud_mram_addr addr = src + i * 8;

        for (unsigned group_nr = 0; group_nr < 8; ++group_nr) {
            volatile uint64_t* line = line_for_group(r, addr, group_nr);
            flush_cache_line(line);
        }
    }

    invoc_memory_fence();

    for (size_t i = 0; i < sz; ++i) {
        vud_mram_addr addr = src + i * 8;

        for (unsigned group_nr = 0; group_nr < 8; ++group_nr) {
            volatile uint64_t* line = line_for_group(r, addr, group_nr);

            uint64_t mat[8];

            mem_to_mat(line, &mat);
            byte_interleave_mat(&mat);

            for (unsigned ci_nr = 0; ci_nr < 8; ++ci_nr) {
                (*tgt)[get_dpu_id(group_nr, ci_nr)][i] = mat[ci_nr];
            }
        }
    }

    invoc_memory_fence();

    for (size_t i = 0; i < sz; ++i) {
        vud_mram_addr addr = src + i * 8;

        for (unsigned group_nr = 0; group_nr < 8; ++group_nr) {
            volatile uint64_t* line = line_for_group(r, addr, group_nr);
            flush_cache_line(line);
        }
    }

    invoc_memory_fence();
}
