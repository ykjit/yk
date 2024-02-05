// ignore-if: test ${YK_ARCH} != "x86_64"
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_PRINT_JITSTATE=1
//   stderr:
//     ...
//     jit-state: enter-jit-code
//     ...
//  stdout:
//     exit

// Check that disassembly-based PT decoding does the right thing with
// zero-length calls.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

extern uintptr_t zero_len_call(void);

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int sum = 0;
  int i = 20;
  NOOPT_VAL(loc);
  NOOPT_VAL(sum);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    sum += zero_len_call();
    i--;
  }
  printf("exit");
  NOOPT_VAL(sum);
  yk_location_drop(loc);
  yk_mt_drop(mt);
  return (EXIT_SUCCESS);
}
