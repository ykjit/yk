// Compiler:
//   env-var: YKD_PRINT_JITSTATE=1
// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   stderr:
//     jit-state: start-tracing
//     i=3
//     --- Begin jit-pre-opt ---
//     ...
//       switch i32 %{{cond}}, label %{{default-bb}} [
//         i32 100, label %guardfail
//         i32 200, label %guardfail
//         i32 300, label %guardfail
//       ]
//
//     guardfail:...
//     ...
//     --- End jit-pre-opt ---
//     jit-state: stop-tracing
//     i=2
//     jit-state: enter-jit-code
//     i=1
//     switch_default: guard-failure

// Check that tracing the default arm of a switch works correctly.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  int loc = 0;
  int i = 3;
  NOOPT_VAL(i);
  while (i > 0) {
    yk_control_point(loc);
    fprintf(stderr, "i=%d\n", i);
    switch (i) {
      case 100:
        fprintf(stderr, "one hundred\n");
      case 200:
        fprintf(stderr, "two hundred\n");
      case 300:
        fprintf(stderr, "three hundred\n");
      default:
        i--;
    }
  }
  abort(); // FIXME: unreachable due to aborting guard failure earlier.

  return (EXIT_SUCCESS);
}
