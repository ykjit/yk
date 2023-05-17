// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_PRINT_JITSTATE=1
//   env-var: YKD_PRINT_IR=aot,jit-post-opt
//   stderr:
//     jit-state: start-tracing
//     y=100
//     jit-state: stop-tracing
//     ...
//     --- Begin aot ---
//     ...
//     @{{fname}} = constant [6 x i8] c"inner\00"
//     ...
//     define dso_local {{size_t}} @inner({{size_t}} noundef %0, {{size_t}} noundef %1)...
//       ...
//     }
//     ...
//     define dso_local i32 @main(...
//       ...
//       call void (ptr, i64, ...) @__yk_record_promote_usize(ptr @{{fname}}, {{size_t}} 1, {{size_t}} 0, {{size_t}} %{{x}})...
//       call void (i64, i32, ...) @llvm.experimental.stackmap(...
//       %{{ret}} = call {{size_t}} @inner({{size_t}} noundef %{{x}}, {{size_t}} noundef %{{y}})...
//       ...
//     }
//     ...
//     --- End aot ---
//     --- Begin jit-post-opt ---
//     ...
//     define ptr @__yk_compiled_trace_0(...
//       ...
//       %{{cond}} = icmp eq i64 {{jitx}}, 100
//       br i1 %{{cond}}, label %{{succbb}}, label %{{failbb}}
//
//     {{succbb}}:...
//       ...
//       br label %{{succbb2}}
//
//     {{failbb}}:...
//       ...
//       %{{deopt}} = call ptr (...) @llvm.experimental.deoptimize...
//       ...
//       ret ...
//
//     {{succbb2}}:...
//       ...
//       %{{res}} = add {{size_t}} %{{arg1}}, 100...
//       ...
//     }
//     ...
//     --- End jit-post-opt ---
//     y=200
//     jit-state: enter-jit-code
//     y=300
//     y=400
//     y=500
//     jit-state: deoptimise
//     jit-state: exit-jit-code
//     y=700
//     jit-state: enter-jit-code
//     y=800
//     y=900
//     y=1000
//     jit-state: deoptimise
//     jit-state: exit-jit-code
//     y=1999

// Check that variable promotions are guarded correctly.

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

#define ELEMS 10

__attribute__((yk_promote("x"))) size_t inner(size_t x, size_t y) {
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
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
