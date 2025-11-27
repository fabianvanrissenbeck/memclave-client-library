#include <vud.h>
#include <vud_mem.h>
#include <vud_ime.h>

#include <stdio.h>
#include <string.h>
#include <assert.h>

#define IME_DPU_CNTR 0x0
#define IME_CLIENT_PUBKEY 0x110
#define IME_DPU_PUBKEY 0x10

int main(void) {
    vud_rank r = vud_rank_alloc(VUD_ALLOC_ANY);

    if (r.err) {
        printf("cannot allocate rank: %s\n", vud_error_str(r.err));
        return -1;
    }

    vud_ime_install_key(&r, (uint8_t[32]) { 0x12, 0x34 }, NULL, NULL);

    if (r.err) {
        printf("cannot exchange keys: %s\n", vud_error_str(r.err));
        return -1;
    }

    puts("Done.");
    vud_rank_free(&r);

    return 0;
}