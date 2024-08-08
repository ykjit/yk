// Compiler:
//   env-var: YKD_LOG_IR=-:jit-pre-opt

// Check that the system is OK with the control point being in a nested loop.

#include <stdbool.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  int outers = 100;
  int inners = 100;
  NOOPT_VAL(outers);
  NOOPT_VAL(inners);
  for (int i = 0; i < outers; i++) {
    for (int j = 0; j < inners; j++) {
      yk_mt_control_point(mt, NULL); // In a nested loop!
    }
  }
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
