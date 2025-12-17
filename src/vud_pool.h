#ifndef VUD_POOL_H
#define VUD_POOL_H

#include <pthread.h>
#include <stdbool.h>

/**
 * @brief called once per pool_do call for each worker
 * @param id ID of the worker - The id is between 0 and the worker count minus 1
 * @param nr_worker number of workers in the current pool
 * @param arg argument passed via pool_do
 */
typedef void (*vud_pool_worker)(unsigned id, unsigned nr_worker, void* arg);

/** a simple thread pool implementation used for parallel memory accesses */
typedef struct vud_pool vud_pool;

/**
 * @brief create a new thread pool
 *
 * This will spawn n_worker - 1 threads in the background, which
 * will immediately block on a barrier waiting for something to do.
 *
 * @param n_worker number of workers in the pool
 * @returns a thread pool on success or NULL on failure
 */
vud_pool* vud_pool_init(unsigned n_worker);

/**
 * @brief perform an operation on all workers of the pool
 * @param pool thread pool to perform operation with
 * @param worker function pointer to the operation to perform
 * @param arg argument passed to all worker threads
 */
void vud_pool_do(vud_pool* pool, vud_pool_worker worker, void* arg);

/**
 * @brief free up all resources (including threads and locks) used for the pool
 */
void vud_pool_free(vud_pool* pool);

#endif
