// ignore-if: test $YK_JIT_COMPILER != "yk" -o "$YKB_TRACER" = "swt"
// Run-time:
//   env-var: YKD_LOG_IR=-:jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=-
//   stderr:
//     jitstate: start-tracing
//     add 5
//     sub 3
//     jitstate: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{1}}: i32 = Add %{{2}}, %{{argc}}
//     ...
//     %{{3}}: i32 = Sub %{{4}}, %{{argc}}
//     ...
//     --- End jit-pre-opt ---
//     add 4
//     sub 2
//     jitstate: enter-jit-code
//     add 3
//     sub 1
//     add 2
//     sub 0
//     jitstate: deoptimise
//     exit

// Test some binary operations.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline)) int foo(int i) { return i + 3; }

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    int add = i + argc;
    int sub = i - argc;
    fprintf(stderr, "add %d\n", add);
    fprintf(stderr, "sub %d\n", sub);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
