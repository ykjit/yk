// ignore-if: test "$YK_JITC" != "j2"
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_LOG=4
//   stderr:
//     6
//     yk-tracing: start-tracing
//     1
//     return
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     return
//     --- End jit-pre-opt ---
//     ...
//     2
//     yk-execution: enter-jit-code
//     1
//     return
//     exit

// Check that traces that left the interpreter loop during tracing emit a
// return instruction.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

void loop(YkMT *, YkLocation *, int);

__attribute__((yk_outline))
void loop(YkMT *mt, YkLocation *loc, int i) {
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, loc);
    fprintf(stderr, "%d\n", i);
    if (i == 6) {
      loop(mt, loc, i - 5);
    }
    i--;
  }
  fprintf(stderr, "return\n");
  return;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 1);
  yk_mt_sidetrace_threshold_set(mt, 2);
  YkLocation loc = yk_location_new();

  loop(mt, &loc, 6);
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
