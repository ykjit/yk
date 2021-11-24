// Compiler:
//   env-var: YKD_PRINT_JITSTATE=1
// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   stderr:
//     jit-state: start-tracing
//     i=3
//     --- Begin jit-pre-opt ---
//     ...
//       %{{cond}} = icmp...
//       br i1 %{{cond}}, label %{{succ-bb}}, label %guardfail
//
//     guardfail:...
//     ...
//     --- End jit-pre-opt ---
//     jit-state: stop-tracing
//     i=2
//     jit-state: enter-jit-code
//     i=1
//     switch: guard-failure

// Check that tracing a non-default switch arm works correctly.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkLocation loc = yk_location_new();
  int i = 3;
  int j = 300;
  NOOPT_VAL(i);
  NOOPT_VAL(j);
  while (i > 0) {
    yk_control_point(&loc);
    fprintf(stderr, "i=%d\n", i);
    switch (j) {
      case 100:
        i = 997;
      case 200:
        i = 998;
      case 300:
        i--;
        break;
      default:
        i = 999;
    }
  }
  abort(); // FIXME: unreachable due to aborting guard failure earlier.

  return (EXIT_SUCCESS);
}
