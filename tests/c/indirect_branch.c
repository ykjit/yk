// # guards for indirect branches not implemented.
// ignore-if: true
// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   env-var: YKD_PRINT_JITSTATE=1
//   stderr:
//     ...

// Check that tracing an `indirectbr` works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  int loc = 0;
  int i = 5;
  int idx = 1;
  void *dispatch[] = {&&label1, &&label3, &&label2};
  NOOPT_VAL(i);
  NOOPT_VAL(idx);
  while (i > 0) {
    yk_mt_control_point(loc);
    fprintf(stderr, "i=%d\n", i);
    goto *dispatch[idx];
  label1:
    abort(); // unreachable.
  label2:
    abort(); // unreachable.
  label3:
    i--;
  }
  abort(); // FIXME: unreachable due to aborting guard failure earlier.

  return (EXIT_SUCCESS);
}
