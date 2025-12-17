#include "vud_pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>

typedef struct vud_pool_args {
    vud_pool* pool;
    vud_pool_worker func;
    void* arg;
    unsigned id;
    volatile bool do_exit;
} vud_pool_args;

struct vud_pool {
    unsigned n_worker;
    pthread_t* workers;
    vud_pool_args* args;
    pthread_barrier_t bar_start;
    pthread_barrier_t bar_end;
    volatile bool failed;
};

static bool pool_thread_iter(vud_pool_args* arg) {
    int err = pthread_barrier_wait(&arg->pool->bar_start);

    if (err && err != PTHREAD_BARRIER_SERIAL_THREAD) {
        perror("cannot wait on barrier");
        arg->pool->failed = true;

        return false;
    }

    if (!arg->do_exit) {
        (arg->func)(arg->id, arg->pool->n_worker, arg->arg);
    } else {
        return false;
    }

    err = pthread_barrier_wait(&arg->pool->bar_end);

    if (err && err != PTHREAD_BARRIER_SERIAL_THREAD) {
        perror("cannot wait on barrier");
        arg->pool->failed = true;

        return false;
    }

    return true;
}

static void* pool_thread_main(void* arg_ptr) {
    vud_pool_args* args = arg_ptr;

    while (pool_thread_iter(args)) {}

    return NULL;

}

vud_pool* vud_pool_init(unsigned n_worker) {
    vud_pool* res = malloc(sizeof(vud_pool));

    if (res == NULL) {
        return NULL;
    }

    *res = (vud_pool) {
        .n_worker = n_worker,
        .args = calloc(n_worker, sizeof(*res->args)),
        .workers = calloc(n_worker - 1, sizeof(*res->workers)), // calling thread is used as a worker as well
    };

    if (pthread_barrier_init(&res->bar_start, NULL, n_worker) != 0) {
        goto failure;
    }

    if (pthread_barrier_init(&res->bar_end, NULL, n_worker) != 0) {
        goto failure_2;
    }

    for (unsigned j = 0; j < n_worker; j++) {
        res->args[j] = (vud_pool_args) {
            .pool = res,
            .id = j
        };
    }

    unsigned i = 1;

    for (; i < n_worker; i++) {
        if (pthread_create(&res->workers[i - 1], NULL, pool_thread_main, &res->args[i]) != 0) {
            goto failure_3;
        }
    }

    return res;

failure_3:
    for (int j = 1; j < i; ++j) {
        pthread_kill(res->workers[j], SIGKILL);
    }

    pthread_barrier_destroy(&res->bar_end);

failure_2:
    pthread_barrier_destroy(&res->bar_start);

failure:

    free(res->args);
    free(res->workers);

    return NULL;
}

void vud_pool_do(vud_pool* pool, vud_pool_worker worker, void* arg) {
    if (pool->failed) { return; }

    for (int i = 0; i < pool->n_worker; i++) {
        pool->args[i].func = worker;
        pool->args[i].arg = arg;
        pool->args[i].do_exit = false;
    }

    pool_thread_iter(&pool->args[0]);
}

void vud_pool_free(vud_pool* pool) {
    if (pool->failed) { return; }

    for (int i = 0; i < pool->n_worker; i++) {
        pool->args[i].do_exit = true;
    }

    pool_thread_iter(&pool->args[0]);

    for (int i = 1; i < pool->n_worker; i++) {
        if (pthread_join(pool->workers[i - 1], NULL) != 0) {
            perror("cannot join pthread");
        }
    }

    free(pool->args);
    free(pool->workers);

    *pool = (vud_pool) { 0 };
}
