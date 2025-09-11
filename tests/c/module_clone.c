// Run-time:
//   env-var: YKD_LOG_IR=aot
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     ...
//     --- Begin aot ---
//     ...
//     func __yk_opt_bar() -> ptr;
//     ...
//     func __yk_opt_foo(%arg0: i32);
//     ...
//     --- End aot ---
//     ...

// A very basic basic smoke test for "module cloning".
//
// At least checks that some functions have been cloned with the __yk_opt
// prefix. Better testing is deferred to the ykllvm repo where we can look
// inside the LLVM AOT IR of the optimised clones.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

__attribute__((noinline))
char *bar() {
  return "bar\n";
}

__attribute__((noinline))
void foo(int i) {
  fputs(bar(), stderr);
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

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%d\n", i);
    foo(i);
    i--;
  }
  fprintf(stderr, "%d\n", i);
  fprintf(stderr, "exit\n");
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
