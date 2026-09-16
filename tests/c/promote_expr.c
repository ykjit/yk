// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   env-var: YKD_LOG_IR=hir
//   stderr:
//     yk-tracing: start-tracing
//     y=50
//     yk-tracing: stop-tracing
//     --- Begin hir ---
//     ...
//     %{{11}}: i64 = 25
//     ...
//     %{{14}}: i1 = icmp eq %{{_}}, %{{11}}
//     guard true, %{{14}}, ...
//     ...
//     --- End hir ---
//     y=100
//     yk-execution: enter-jit-code {"trid": "0"}
//     y=150
//     y=200
//     y=250
//     yk-execution: deoptimise {"trid": "0", "gidx": "0"}

// Check that expression promotion works in traces.

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((__noinline__)) size_t inner(size_t x) {
  yk_promote(x + 25);
  return x + 25;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  size_t x = 25;
  size_t y = 0;
  NOOPT_VAL(x);

  for (int i = 0; i < 5; i++) {
    yk_mt_control_point(mt, &loc);
    y += inner(x);
    fprintf(stderr, "y=%" PRIu64 "\n", y);
  }

  NOOPT_VAL(y);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
