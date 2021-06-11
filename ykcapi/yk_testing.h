// Functions exported only for testing.

#include <stdint.h>

#define SW_TRACING 0
#define HW_TRACING 1

void *__yktrace_hwt_mapper_blockmap_new(void);
size_t __yktrace_hwt_mapper_blockmap_len(void *mapper);
void __yktrace_hwt_mapper_blockmap_free(void *mapper);

// Until we have a proper API for tracing, variables that we want to pass into
// a compiled trace need to be "registered" by passing them into
// __yktrace_start_tracing. While the start tracing call ignores them, it
// allows us identify them when preparing the inlined trace code.
void *__yktrace_start_tracing(uintptr_t kind, ...);
void *__yktrace_stop_tracing(void *tt);
size_t __yktrace_irtrace_len(void *trace);
void __yktrace_irtrace_get(void *trace, size_t idx, char **res_func,
                           size_t *res_bb);
void *__yktrace_irtrace_compile(void *trace);
void __yktrace_drop_irtrace(void *trace);
