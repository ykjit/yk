// ignore-if: test $YK_JIT_COMPILER != "yk" -o "$YKB_TRACER" = "swt"
// Run-time:
//   env-var: YKD_LOG_IR=-:aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=-
//   stderr:
//     jitstate: start-tracing
//     i=3
//     jitstate: stop-tracing
//     --- Begin aot ---
//     ...
//     switch 300i32, bb{{bb14}}, [299 -> bb{{bb10}}, 298 -> bb{{bb11}}] [safepoint: 0i64, (%0_2, %0_3, %0_5, %0_6, %1_2)]
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{18}}: i1 = eq 300i32, 299i32
//     %{{19}}: i1 = eq 300i32, 298i32
//     %{{20}}: i1 = or %{{21}}, %{{22}}
//     guard %{{20}}, false
//     ...
//     --- End jit-pre-opt ---
//     i=2
//     jitstate: enter-jit-code
//     i=1
//     jitstate: deoptimise
//     ...

// Check that tracing the default arm of a switch works correctly.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();
  int i = 3;
  int j = 300;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "i=%d\n", i);
    switch (j) {
    case 299:
      exit(1);
    case 298:
      exit(1);
    default:
      i--;
      break;
    }
  }
  yk_location_drop(loc);
  yk_mt_drop(mt);

  return (EXIT_SUCCESS);
}
