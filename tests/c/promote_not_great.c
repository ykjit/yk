// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=-
//   env-var: YKD_LOG_IR=-:jit-post-opt
//   stderr:
//     jitstate: start-tracing
//     z=2
//     jitstate: stop-tracing
//     --- Begin jit-post-opt ---
//     ...
//     define ptr @__yk_compiled_trace_0(...
//       ...
//       %{{cond1}} = icmp eq i64 %{{x1}}, 200...
//       ...
//       %{{cond2}} = icmp eq i64 %{{x2}}, 200...
//       ...
//     }
//     ...
//     --- End jit-post-opt ---
//     z=4
//     jitstate: enter-jit-code
//     z=6
//     z=8
//     z=10
//     jitstate: deoptimise

// Demonstrate where yk_promote() needs improvement.

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

size_t inner(size_t x, size_t y) {
  yk_promote(x);
  yk_promote(y);
  size_t ret = x / y; // JIT doesn't manage to promote `x`.
  yk_promote(x);      // JIT doesn't manage to kill repeated guard.
  return ret;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  size_t x = 200, y = 100, z = 0;
  NOOPT_VAL(x);
  NOOPT_VAL(y);

  for (int i = 0; i < 5; i++) {
    yk_mt_control_point(mt, &loc);
    z += inner(x, y);
    fprintf(stderr, "z=%" PRIu64 "\n", z);
  }

  NOOPT_VAL(y);
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
