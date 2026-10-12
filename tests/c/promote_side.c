// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   env-var: YKD_LOG_IR=hir
//   stderr:
//     yk-tracing: start-tracing
//     101
//     yk-tracing: stop-tracing
//     --- Begin hir ---
//     ...
//     %{{_}}: i32 = 101
//     ...
//     --- End hir ---
//     101
//     yk-execution: enter-jit-code {"trid": "0"}
//     101
//     yk-execution: deoptimise {"trid": "0", "gidx": "{{_}}"}
//     202
//     yk-execution: enter-jit-code {"trid": "0"}
//     yk-execution: deoptimise {"trid": "0", "gidx": "{{_}}"}
//     yk-tracing: start-side-tracing
//     202
//     yk-tracing: stop-tracing
//     --- Begin hir ---
//     ...
//     %{{new}}: i32 = 2
//     %{{cmp}}: i1 = icmp eq %{{_}}, %{{new}}
//     guard true, %{{cmp}}, ...
//     ...
//     %{{_}}: i32 = 202
//     ...
//     --- End hir ---
//     202
//     yk-execution: enter-jit-code {"trid": "0"}
//     202
//     yk-execution: deoptimise {"trid": "1", "gidx": "{{_}}"}

// Check that a guard failing from a `yk_promote` creates a side-trace that is
// specialised to the newly-seen value.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(void) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  yk_mt_sidetrace_threshold_set(mt, 2);
  YkLocation loc = yk_location_new();
  int values[] = {1, 1, 1, 2, 2, 2, 2};
  int n = 7;
  NOOPT_VAL(values);
  NOOPT_VAL(n);
  for (int i = 0; i < n; i++) {
    yk_mt_control_point(mt, &loc);
    int x = yk_promote(values[i]);
    fprintf(stderr, "%d\n", x * 101);
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return EXIT_SUCCESS;
}
