// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=hir
//   env-var: YKD_LOG=4
//   stderr:
//     5: 0 0
//     yk-tracing: start-tracing
//     4: 1 1
//     yk-tracing: stop-tracing
//     --- Begin hir ---
//     ...
//     %{{_}}: i32 = atomicload relaxed %{{_}}
//     ...
//     %{{_}}: i32 = atomicload acquire %{{_}}
//     ...
//     --- End hir ---
//     3: 2 2
//     yk-execution: enter-jit-code {"trid": "0"}
//     2: 3 3
//     1: 4 4
//     yk-execution: deoptimise {"trid": "0", "gidx": "{{_}}"}
//     exit

// Check that atomic loads are translated into HIR correctly. Note: this test
// isn't trying to do anything clever across threads.

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

static _Atomic int atint;

// FIXME: we don't currently implement atomic stores.
__attribute__((yk_outline)) static void update(int v) {
  atomic_store_explicit(&atint, v, memory_order_release);
}

int main(void) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 1);
  YkLocation loc = yk_location_new();
  int i = 5;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    int x = atomic_load_explicit(&atint, memory_order_relaxed);
    int y = atomic_load_explicit(&atint, memory_order_acquire);
    fprintf(stderr, "%d: %d %d\n", i, x, y);
    update(x + 1);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  fprintf(stderr, "exit");
  return EXIT_SUCCESS;
}
