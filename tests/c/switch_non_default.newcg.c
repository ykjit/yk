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
//     switch ${{10_1}}, bb{{bb14}}, [300 -> bb{{bb11}}, 299 -> bb{{bb12}}]
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{cond}}: i8 = Icmp %{{20}}, Equal, 300i32
//     Guard %{{cond}}, true
//     ...
//     --- End jit-pre-opt ---
//     i=2
//     jitstate: enter-jit-code
//     i=1
//     jitstate: deoptimise
//     ...

// Check that tracing a non-default switch arm works correctly.

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
  NOOPT_VAL(j);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "i=%d\n", i);
    switch (j) {
    case 300: // This case is traced.
      i--;
      break;
    case 299:
      exit(1);
    default:
      exit(1);
    }
  }
  yk_location_drop(loc);
  yk_mt_drop(mt);

  return (EXIT_SUCCESS);
}
