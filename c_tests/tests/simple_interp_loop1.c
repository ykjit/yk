// Compiler:
//   env-var: YKD_PRINT_JITSTATE=1
// Run-time:
//   env-var: YKD_PRINT_IR=aot,jit-pre-opt
//   stderr:
//     jit-state: start-tracing
//     ...
//     define internal %YkCtrlPointVars @__yk_compiled_trace_0(%YkCtrl...
//        ...
//     }
//     ...
//     jit-state: stop-tracing
//     0
//     1
//     2
//     3
//     jit-state: enter-jit-code
//     0
//     1
//     2
//     3
//     jit-state: exit-jit-code
//     0
//     1
//     2
//     3
//     4
//     5

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

// The sole mutable memory cell of the interpreter.
int mem = 9;

// The bytecodes accepted by the interpreter.
#define DEC 1
#define RESTART_IF_NOT_ZERO 2

int main(int argc, char **argv) {
  // A hard-coded program to execute.
  int prog[] = {1, 1, 1, 2, 1, 1};
  size_t prog_len = sizeof(prog) / sizeof(prog[0]);

  // The program counter (FIXME: also serving as a location ID for now).
  int pc = 0;

  NOOPT_VAL(prog);
  NOOPT_VAL(prog_len);
  NOOPT_VAL(pc);
  NOOPT_VAL(mem);

  // interpreter loop.
  while (true) {
    yk_control_point(pc);
    if (pc >= prog_len) {
      exit(0);
    }
    int bc = prog[pc];
    fprintf(stderr, "%d\n", pc);
    switch (bc) {
    case DEC:
      mem--;
      pc++;
      break;
    case RESTART_IF_NOT_ZERO:
      if (mem > 0)
        pc = 0;
      else
        pc++;
      break;
    default:
      abort();
    }
  }

  NOOPT_VAL(pc);
  assert(pc == 5);
  return (EXIT_SUCCESS);
}
