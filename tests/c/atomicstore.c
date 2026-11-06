// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=hir
//   env-var: YKD_LOG=4
//   stderr:
//     5: 5 6
//     yk-tracing: start-tracing
//     4: 4 5
//     yk-tracing: stop-tracing
//     --- Begin hir ---
//     ...
//     atomicstore relaxed %{{_}}, %{{_}}
//     ...
//     atomicstore release %{{_}}, %{{_}}
//     ...
//     --- End hir ---
//     3: 3 4
//     yk-execution: enter-jit-code {"trid": "0"}
//     2: 2 3
//     1: 1 2
//     yk-execution: deoptimise {"trid": "0", "gidx": "{{_}}"}
//     exit

// Check that atomic stores are translated into HIR correctly. Note: this test
// isn't trying to do anything clever across threads.

#include <inttypes.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

static _Atomic int at_int;

int main(void) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 1);
  YkLocation loc = yk_location_new();
  int i = 5;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    atomic_store_explicit(&at_int, i, memory_order_relaxed);
    uint64_t x = atomic_load_explicit(&at_int, memory_order_relaxed);
    atomic_store_explicit(&at_int, i + 1, memory_order_release);
    uint64_t y = atomic_load_explicit(&at_int, memory_order_acquire);
    fprintf(stderr, "%d: %" PRIu64 " %" PRIu64 "\n", i, x, y);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  fprintf(stderr, "exit");
  return EXIT_SUCCESS;
}
