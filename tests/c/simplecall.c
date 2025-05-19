// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O1
// Run-time:
//   env-var: YKD_LOG_IR=aot,jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     4
//     foo
//     yk-tracing: stop-tracing
//     --- Begin aot ---
//     ...
//     func main(%arg0: i32, %arg1: ptr) -> i32 {
//     ...
//     --- End aot ---
//     --- Begin jit-pre-opt ---
//     ...
//     %{{1}}: i1 = sgt %{{2}}, 1i32
//     ...
//     %{{3}}: i64 = call @fwrite(%{{4}}, 4i64, 1i64, %{{5}})
//     ...
//     --- End jit-pre-opt ---
//     3
//     foo
//     yk-execution: enter-jit-code
//     2
//     foo
//     1
//     bar
//     yk-execution: deoptimise
//     0
//     exit

// Check that call inlining works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

void foo(int i) {
  if (i > 1) {
    fputs("foo\n", stderr);
  } else {
    fputs("bar\n", stderr);
  }
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int res = 9998;
  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%d\n", i);
    foo(i);
    res += 2;
    i--;
  }
  fprintf(stderr, "%d\n", i);
  fprintf(stderr, "exit\n");
  NOOPT_VAL(res);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
