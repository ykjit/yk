// ignore-if: test "$YK_JITC" = "j2"
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   env-var: YKD_LOG_IR=aot
//   stderr:
//     yk-tracing: start-tracing
//     x=200
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     %{{_}}: i64 = promote %{{_}} ...
//     ...
//     --- End aot ---
//     x=300
//     yk-execution: enter-jit-code
//     ...

// Check that the ConditionalPromoteCalls pass:
// 1. The tracing state check is NOT serialised into AOT IR (it's skipped)
// 2. The promote call IS present in AOT IR

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  size_t x = 100;
  NOOPT_VAL(x);

  for (int i = 0; i < 5; i++) {
    yk_mt_control_point(mt, &loc);
    x = yk_promote(x);
    x += 100;
    fprintf(stderr, "x=%zu\n", x);
  }

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
