#ifndef _COMMON_H_
#define _COMMON_H_

// Structures used by both the host and the dpu to communicate information 
typedef struct {
    uint32_t size;
	enum kernels {
	    kernel1 = 0,
	    nr_kernels = 1,
	} kernel;
} dpu_arguments_t;

typedef struct {
    uint64_t cycles;
} dpu_results_t;

// Transfer size between MRAM and WRAM
#ifdef BL
#define BLOCK_SIZE_LOG2 BL
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#else
#define BLOCK_SIZE_LOG2 8
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#define BL BLOCK_SIZE_LOG2
#endif

//#define UINT32
#define UINT64

// Data type
#if defined(UINT32)
  #define T uint32_t
#elif defined(UINT64)
  #define T uint64_t
#elif defined(INT32)
  #define T int32_t
#elif defined(INT64)
  #define T int64_t
#elif defined(FLOAT)
  #define T float
#elif defined(DOUBLE)
  #define T double
#else
  #error "No data type defined for T"
#endif

//#define PERF 1 // Use perfcounters?
#define PRINT 0

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#endif
