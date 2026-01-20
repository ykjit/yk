// ignore-if: test "$YK_JITC" != "j2"
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   env-var: YKD_LOG_IR=jit-pre-opt
//   stderr:
//     yk-tracing: start-tracing
//     y=100
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     ; guard 0
//     %0: ptr = arg
//     %1: ptr = arg
//     %2: ptr = arg
//     %3: ptr = arg
//     %4: ptr = arg
//     %5: i64 = arg
//     %6: i64 = arg
//     %7: ptr = arg
//     %8: ptr = arg
//     %9: i64 = arg
//     term [%0, %1, %2, %3, %4, %5, %6, %7, %7, %8, %9, %9]
//     ...
//     --- End jit-pre-opt ---
//     y=200
//     yk-execution: enter-jit-code
//     y=300
//     y=400
//     y=500
//     yk-execution: deoptimise ...
//     y=700
//     yk-execution: enter-jit-code
//     y=800
//     y=900
//     y=1000
//     yk-execution: deoptimise ...
//     y=1999

// Check that repeated instruction indexes in guards work correctly: note the
// repeated `%7`s and `%9`s in the `term` instruction above.

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

#define ELEMS 10

size_t inner(size_t x, size_t y) {
  yk_promote(x);
  y += x;
  return y;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  // We will trace 100 baked into the trace, and every iteration where there's
  // a 200 we should guard fail.
  size_t xs[ELEMS] = {100, 100, 100, 100, 100, 200, 100, 100, 100, 999};
  size_t y = 0;
  NOOPT_VAL(xs);

  // On the third iteration, a guard must fail to stop us blindly adding 100
  // instead of 200.
  for (int i = 0; i < ELEMS; i++) {
    yk_mt_control_point(mt, &loc);
    y = inner(xs[i], y);
    fprintf(stderr, "y=%" PRIu64 "\n", y);
  }

  NOOPT_VAL(y);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
