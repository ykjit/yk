// Run-time:
//   env-var: YKD_LOG_IR=-:jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YK_LOG=4
//   stderr:
//     yk-jit-event: start-tracing
//     foo 7
//     yk-jit-event: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     %{{1}}: i32 = icall %{{2}}(%{{3}})
//     ...
//     --- End jit-pre-opt ---
//     foo 6
//     yk-jit-event: enter-jit-code
//     foo 5
//     foo 4
//     yk-jit-event: deoptimise
//     exit

// Check that indirect calls work.

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
  int (*fn)(int) = foo;

  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  NOOPT_VAL(fn);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    int x = fn(i);
    fprintf(stderr, "foo %d\n", x);
    i--;
  }
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
