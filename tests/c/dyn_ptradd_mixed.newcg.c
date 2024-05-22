// ignore-if: test $YK_JIT_COMPILER != "yk" -o "$YKB_TRACER" = "swt"
// Run-time:
//   env-var: YKD_LOG_IR=-:aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=-
//   stderr:
//     jitstate: start-tracing
//     i=4, y=7
//     jitstate: stop-tracing
//     --- Begin aot ---
//     ...
//     ${{9_4}}: ptr = PtrAdd @line, 4 + (${{9_3}} * 8)
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{14}}: ptr = ptr_add %{{13}}, 4i32
//     %{{15}}: i64 = mul %{{12}}, 8i64
//     %{{16}}: ptr = ptr_add %{{14}}, %{{15}}
//     ...
//     --- End jit-pre-opt ---
//     i=3, y=6
//     jitstate: enter-jit-code
//     i=2, y=5
//     i=1, y=4
//     jitstate: deoptimise

// Check dynamic ptradd instructions work.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

struct point {
  uint32_t x;
  uint32_t y;
};

struct point line[] = {
    {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7},
};

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "i=%d, y=%d\n", i, line[i].y);
    i--;
  }
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
