// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_PRINT_JITSTATE=1
//   env-var: YKD_PRINT_IR=jit-post-opt
//   stderr:
//     jit-state: start-tracing
//     y=50
//     jit-state: stop-tracing
//     --- Begin jit-post-opt ---
//     ...
//     define ptr @__yk_compiled_trace_0(...
//       ...
//       %{{cond}} = icmp eq i64 {{x}}, 50
//       br i1 %{{cond}}, label %{{succbb}}, label %{{failbb}}
//
//     {{failbb}}:...
//       ...
//       %{{deopt}} = call ptr (...) @llvm.experimental.deoptimize...
//       ret ...
//
//     {{succbb}}:...
//       ...
//       %{{res}} = add {{size_t}} %{{arg1}}, 50...
//       ...
//     }
//     ...
//     --- End jit-post-opt ---
//     y=100
//     jit-state: enter-jit-code
//     y=150
//     y=200
//     y=250
//     jit-state: deoptimise

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
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
