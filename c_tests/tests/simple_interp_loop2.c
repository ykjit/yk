// Compiler:
//   env-var: YKD_PRINT_JITSTATE=1
// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   stderr:
//     jit-state: start-tracing
//     ...
//     define internal %YkCtrlPointVars @__yk_compiled_trace_0(%YkCtrlPointVars %0) {
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
//     ...
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

// The sole mutable memory cell of the interpreter.
int mem = 3;

// The bytecodes accepted by the interpreter.
#define NOP 0
#define DEC 1
#define RESTART_IF_NOT_ZERO 2
#define EXIT 3

int main(int argc, char **argv) {
  // A hard-coded program to execute.
  int prog[] = {0, 0, 1, 2, 0, 3};
  size_t prog_len = sizeof(prog) / sizeof(prog[0]);

  // The program counter (FIXME: also serving as a location ID for now).
  int pc = 0;

  NOOPT_VAL(pc);
  NOOPT_VAL(prog);
  NOOPT_VAL(prog_len);
  NOOPT_VAL(mem);

  // interpreter loop.
  while (true) {
    yk_control_point(pc);
    assert(pc < prog_len);
    int bc = prog[pc];
    fprintf(stderr, "%d\n", pc);
    switch (bc) {
    case NOP:
      pc++;
      break;
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
    case EXIT:
      goto done;
    default:
      abort(); // unreachable.
    }
  }
done:
  NOOPT_VAL(pc);
  assert(pc == 5);
  return (EXIT_SUCCESS);
}
