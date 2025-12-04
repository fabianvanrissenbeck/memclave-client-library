#include <vud.h>
#include <vud_sk.h>
#include <vud_mem.h>
#include <vud_ime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

int main(void) {
    uint64_t* buf = malloc(128 << 10);
    assert(buf != NULL);

    long sz = vud_sk_from_elf("../add", 128 << 10, buf);

    if (sz < 0) {
        puts("cannot load subkernel");
        return -1;
    }

    uint8_t key[32];

    for (int i = 0; i < 32; ++i) { key[i] = 0x80 + i; }

    int res = vud_enc_auth_sk(buf, key);

    if (res < 0) {
        puts("cannot encrypt subkernel");
        return -1;
    }

    FILE* fp = fopen("tmp.sk", "wb");
    assert(fp != NULL);

    fwrite(buf, 1, sz, fp);
    fclose(fp);

    free(buf);
    return 0;
}