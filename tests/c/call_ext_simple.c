// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O1
// Run-time:
//   env-var: YKD_LOG_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   stderr:
//     ...
//     %{{7}}: i32 = call %{{_}}(%{{_}}, %{{_}}) ; @putc
//     ...
//   stdout:
//     12

// Check that calling an external function works.

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

  int ch = '1';
  NOOPT_VAL(ch);
  while (ch != '3') {
    yk_mt_control_point(mt, &loc);
    // Note that sometimes the compiler will make this a call to putc(3).
    putchar(ch);
    ch++;
  }
  fflush(stdout);

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
