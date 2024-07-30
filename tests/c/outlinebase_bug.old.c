// Run-time:
//   env-var: YKD_LOG_IR=-:jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG_JITSTATE=-
//   stderr:
//     ...
//     --- Begin jit-pre-opt ---
//     ...
//     %{{1}} = call i32 @fputc(...
//     ...
//     %{{2}} = call i32 @fputc(...
//     ...
//     %{{3}} = call i32 @fputc(...
//     ...
//     %{{4}} = call i32 @fputc(...
//     ...
//     --- End jit-pre-opt ---
//     ...
//   stdout:
//     exit

// Test that outlining works as expected. Each trace in this test does two
// rounds of the main interpreter loop, resulting in four `fprintf`s being
// copied into the JIT module. However, if we forgot to reset the `OutlineBase`
// in `jitmodbuilder.cc` after resetting `Outlining` to `false`, we will
// continue outlining after seeing the first `yk_mt_control_point`, and miss
// the second round of the loop.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int foo(int argc) {
  int sum = 0;
  // the loop ensures this function is outlined.
  while (argc > 0) {
    argc--;
    sum = sum + argc;
  }
  return sum;
}

int bar(int argc) { return foo(argc); }

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int res = 9998;
  int i = 6;
  NOOPT_VAL(loc);
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  while (i > 0) {
    int r = bar(argc + 2);
    NOOPT_VAL(r);
    YkLocation *tmp;
    if (i % 2 == 0) {
      tmp = &loc;
    } else {
      tmp = NULL;
    }
    yk_mt_control_point(mt, tmp);
    fprintf(stderr, "a");
    fprintf(stderr, "a");
    res += 2;
    i--;
  }
  printf("exit");
  NOOPT_VAL(res);
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
