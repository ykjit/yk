// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O0 -Xclang -disable-O0-optnone -Xlinker --lto-newpm-passes=instcombine<max-iterations=1;no-verify-fixpoint>
// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt
//   env-var: YKD_LOG=4
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     yk-tracing: start-tracing
//     i=4, val=6
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     %{{23_0}}: i32 = phi bb{{bb22}} -> 100i32, bb{{bb21}} -> 6i32
//     ...
//     %{{24_0}}: i32 = phi bb{{bb23}} -> %{{23_0}}, bb{{bb19}} -> 3i32
//     ...
//     %{{25_0}}: i32 = phi bb{{bb24}} -> %{{24_0}}, bb{{bb17}} -> 2i32
//     ...
//     %{{26_0}}: i32 = phi bb{{bb25}} -> %{{25_0}}, bb{{bb15}} -> 1i32
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{26}}: i32 = 6
//     ...
//     %{{_}}: i32 = call %{{_}}(%{{_}}, %{{_}}, %{{_}}, %{{26}}) ; @fprintf
//     ...
//     --- End jit-pre-opt ---
//     i=3, val=6
//     yk-execution: enter-jit-code
//     i=2, val=6
//     i=1, val=6
//     yk-execution: deoptimise ...

// Check that PHI nodes JIT properly.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int val = 0;
  int cond = -3;
  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(val);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    NOOPT_VAL(cond);
    if (cond > 0) {
      val = 1;
    } else if (cond == -1) {
      val = 2;
    } else if (cond == -2) {
      val = 3;
    } else if (cond == -3) {
      val = 6;
    } else {
      val = 100;
    }
    fprintf(stderr, "i=%d, val=%d\n", i, val);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
