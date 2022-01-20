// Compiler:
//   env-var: YKD_PRINT_IR=jit-pre-opt

// Check that the system is OK with the control point being in a nested loop.

#include <stdlib.h>
#include <stdbool.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  int outers = 100;
  int inners = 100;
  NOOPT_VAL(outers);
  NOOPT_VAL(inners);
  for (int i = 0; i < outers; i++) {
    for (int j = 0; j < inners; j++) {
      yk_control_point(NULL); // In a nested loop!
    }
  }
  return (EXIT_SUCCESS);
}
