// ignore-if: test "$YKB_TRACER" = "swt"
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_OPT=1
//   env-var: YK_LOG=4
//   env-var: YKD_LOG_IR=-:aot,jit-pre-opt
//   stderr:
//     yk-jit-event: start-tracing
//     y=100
//     yk-jit-event: stop-tracing
//     --- Begin aot ---
//     ...
//     %{{_}}: i64 = promote %{{_}} [safepoint: 0i64, (%{{0_0}}, %{{0_1}})]
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{1}}: i1 = eq %{{_}}, 100i64
//     guard true, %{{1}}, ...
//     %{{_}}: i64 = add 100i64, %{{_}}
//     ...
//     --- End jit-pre-opt ---
//     y=200
//     yk-jit-event: enter-jit-code
//     y=300
//     y=400
//     y=500
//     yk-jit-event: deoptimise

// Check that promotion works in traces.

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

size_t inner(size_t x, size_t y) {
  x = yk_promote(x);
  y += x;
  return y;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  size_t x = 100;
  size_t y = 0;
  NOOPT_VAL(x);

  for (int i = 0; i < 5; i++) {
    yk_mt_control_point(mt, &loc);
    y = inner(x, y);
    fprintf(stderr, "y=%" PRIu64 "\n", y);
  }

  NOOPT_VAL(y);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
