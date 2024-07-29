// Run-time:
//   env-var: YKD_LOG_IR=-:jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=-
//   stderr:
//     jitstate: start-tracing
//     4:0
//     jitstate: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     ...call i32 @f(...
//     ...
//     --- End jit-pre-opt ---
//     3:0
//     jitstate: enter-jit-code
//     2:0
//     1:0
//     jitstate: deoptimise
//     ...

// Check that recursive function calls are not unrolled.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline)) int f(int num) {
  NOOPT_VAL(num);
  if (num == 0)
    return 0;
  else
    return f(num - 1);
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    int x = 3;
    NOOPT_VAL(x);
    fprintf(stderr, "%d:%d\n", i, f(x));
    i--;
  }

  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
