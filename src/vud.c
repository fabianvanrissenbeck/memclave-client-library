#include "vud.h"
#include "common/vci-msg.h"

#include <stdio.h>
#include <errno.h>
#include <fcntl.h>
#include <assert.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/mman.h>

#define RANK_MAP_SIZE 0x200000000
#define RANK_CI_OFFSET 0x20000

static inline vud_rank new_error(vud_error err) {
    return (vud_rank) {
        .err = err,
        .fd = -1,
        .base = MAP_FAILED
    };
}

__attribute__((noinline))
static uint64_t mem_read_64(const volatile uint64_t* src) {
    volatile uint64_t res = 0;

    __asm__ volatile(
        "mfence\n"
        "clflushopt 0(%0)\n"
        "mfence\n"
        "movq 0(%0), %%rax\n"
        "movq %%rax, 0(%1)\n"
        "mfence\n"
        "clflushopt 0(%0)\n"
        "mfence\n"
        :
        : "r" (src), "r" (&res)
        : "rax"
    );

    return res;
}

__attribute__((noinline))
static void mem_write_64(uint64_t v, volatile uint64_t* dest) {
    __asm__ volatile(
        "movq %0, 0(%1)\n"
        "mfence\n"
        :
        : "r" (v), "r" (dest)
        : "memory"
    );
}

/**
 * @brief write to a memory location and wait for it to be changed by another party
 * @param value value to write
 * @param addr address to write to
 * @param out_timeout indicator, whether the wait loop timed out
 */
__attribute__((noinline))
static uint64_t commit_and_wait(uint64_t value, volatile uint64_t* addr, bool* out_timeout) {
    uint64_t timeout = 1024;
    uint64_t limit = UINT32_MAX;
    uint64_t res = 0;

    mem_write_64(value, addr);

    do {
        res = mem_read_64(addr);
        usleep(timeout);

        timeout = timeout << 1 | 1;
    } while (timeout < limit && res == value);

    *out_timeout = timeout >= limit;
    return res;
}

static volatile uint64_t* get_ci_addr(vud_rank* rank) {
    return (volatile uint64_t*)((uintptr_t) rank->base + RANK_CI_OFFSET);
}

vud_rank vud_rank_alloc(int rank_nr) {
    assert(rank_nr >= -1 && rank_nr <= 39);
    assert(("feature not implemented", rank_nr != -1));

    char path[120];
    snprintf(path, 120, "/dev/vpim%d", rank_nr);

    int fd = open(path, O_RDWR, 0);

    if (fd < 0) {
        vud_error err;

        if (errno == EBUSY) {
            err = VUD_RANK_BUSY;
        } else if (errno == ENOENT) {
            err = VUD_NOT_FOUND;
        } else {
            err = VUD_SYSTEM_ERR;
        }

        return new_error(err);
    }

    volatile void* ptr = mmap(NULL, RANK_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    if (ptr == MAP_FAILED) {
        close(fd);
        return new_error(VUD_MEMORY_ERR);
    }

    return (vud_rank) {
        .base = ptr,
        .fd = fd,
        .err = VUD_OK
    };
}

void vud_rank_free(vud_rank* rank) {
    if (rank->base != MAP_FAILED) {
        munmap((void*) rank->base, RANK_MAP_SIZE);
    }

    if (rank->fd >= 0) {
        close(rank->fd);
    }

    rank->base = MAP_FAILED;
    rank->fd = -1;
}

uint8_t vud_rank_qry_mux(vud_rank* rank) {
    if (rank->err) { return 0x0; }

    vci_msg msg = {
        .type = VCI_QRY_MUX,
    };

    bool timeout = false;
    uint64_t res = commit_and_wait(vci_msg_to_qword(msg), get_ci_addr(rank), &timeout);

    if (timeout) {
        rank->err = VUD_CI_TIMEOUT;
        return 0x0;
    }

    vci_msg rsp = vci_msg_from_qword(res);

    if (rsp.type != VCI_QRY_RES) {
        rank->err = VUD_INVALID_RES;
        return 0x0;
    }

    return rsp.n_faulted;
}

void vud_rank_rel_mux(vud_rank* rank) {
    if (rank->err) { return; }

    vci_msg msg = {
        .type = VCI_REL_MUX,
    };

    bool timeout = false;
    uint64_t res = commit_and_wait(vci_msg_to_qword(msg), get_ci_addr(rank), &timeout);

    if (timeout) {
        rank->err = VUD_CI_TIMEOUT;
        return;
    }

    vci_msg rsp = vci_msg_from_qword(res);

    if (rsp.type != VCI_OK) {
        rank->err = VUD_INVALID_RES;
    }
}
