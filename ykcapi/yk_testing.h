// Functions exported only for testing.

#include <stdint.h>
#include <stdio.h>

#define SW_TRACING 0
#define HW_TRACING 1

void *__yktrace_hwt_mapper_blockmap_new(void);
size_t __yktrace_hwt_mapper_blockmap_len(void *mapper);
void __yktrace_hwt_mapper_blockmap_free(void *mapper);

// Blocks the compiler from optimising the specified value or expression.
//
// This is similar to the non-const variant borrowed from Google benchmark:
// https://github.com/google/benchmark/blob/e451e50e9b8af453f076dec10bd6890847f1624e/include/benchmark/benchmark.h#L350
//
// Our version works on a value, rather than a pointer.
//
// Note that Google Benchmark also defines a variant for constant data. At the
// time of writing, NOOPT_VAL seems to suffice (even for constants), but we may
// need to consider using the const version later.
#ifdef __clang__
#define NOOPT_VAL(X) asm volatile("" : "+r,m"(X) : : "memory");
#else
#error non-clang compilers are not supported.
#endif

// Tries to block optimisations by telling the compiler that all memory
// locations are touched. `NOOPT_VAL` is preferred, but you may not always have
// direct access to the value(s) or expression(s) that you wish to block
// optimisations to.
//
// Borrowed from:
// https://github.com/google/benchmark/blob/ab74ae5e104f72fa957c1712707a06a781a974a6/include/benchmark/benchmark.h#L359
#define CLOBBER_MEM() asm volatile("" : : : "memory");

// Stuff for the hwtracer_ykpt suite.
void *__hwykpt_start_collector(void);
void *__hwykpt_stop_collector(void *tc);
void __hwykpt_libipt_vs_ykpt(void *trace);
