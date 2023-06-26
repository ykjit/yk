// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_PRINT_JITSTATE=1
//   env-var: YKD_PRINT_IR=jit-post-opt
//   stderr:
//     jit-state: start-tracing
//     y=1
//     jit-state: stop-tracing
//     --- Begin jit-post-opt ---
//     ...
//     define ptr @__yk_compiled_trace_0(...
//       ...
//       %{{res}} = udiv {{size_t}} 100, %{{notconst}}...
//       ...
//     }
//     ...
//     --- End jit-post-opt ---
//     y=2
//     jit-state: enter-jit-code
//     y=3
//     y=4
//     y=5
//     jit-state: deoptimise

// Check that the incoming argument to `yk_promote()` isn't what is promoted
// (it's the returned value that is!)

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

size_t inner(size_t x) {
  yk_promote(x);  // Didn't capture and use return value!
  return 100 / x; // <- This use of `x` is therefore not promoted.
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  size_t x = 100, y = 0;
  NOOPT_VAL(x);

  for (int i = 0; i < 5; i++) {
    yk_mt_control_point(mt, &loc);
    y += inner(x);
    fprintf(stderr, "y=%" PRIu64 "\n", y);
  }

  NOOPT_VAL(y);
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
