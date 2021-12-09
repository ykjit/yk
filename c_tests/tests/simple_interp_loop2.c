// Compiler:
//   env-var: YKD_PRINT_JITSTATE=1
// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   stderr:
//     jit-state: start-tracing
//     pc=0, mem=3
//     pc=1, mem=3
//     pc=2, mem=3
//     pc=3, mem=2
//     ...
//     define internal %YkCtrlPointVars @__yk_compiled_trace_0(%YkCtrlPointVars %0) {
//       ...
//       %{{x}} = extractvalue %YkCtrlPointVars...
//       ...
//     {{guard-fail-bb}}:...
//       ...
//       call void (i32, i8*, ...) @errx(i32 0,...
//       unreachable
//
//     {{another-bb}}:...
//       ...
//       %{{restart-cond}} = icmp sgt i32 %{{mem}}, 0...
//       br i1 %{{restart-cond}}, label %{{restart-bb}}, label %{{guard-fail-bb}}
//
//     {{restart-bb}}:...
//       ...
//       ret %YkCtrlPointVars %{{ret}}
//     }
//     ...
//     jit-state: stop-tracing
//     pc=0, mem=2
//     pc=1, mem=2
//     pc=2, mem=2
//     pc=3, mem=1
//     jit-state: enter-jit-code
//     pc=0, mem=1
//     pc=1, mem=1
//     pc=2, mem=1
//     pc=3, mem=0
//     simple_interp_loop2: guard-failure
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
  int prog[] = {NOP, NOP, DEC, RESTART_IF_NOT_ZERO, NOP, EXIT};
  size_t prog_len = sizeof(prog) / sizeof(prog[0]);

  // Create one location for each potential PC value.
  YkLocation locs[prog_len];
  for (int i = 0; i < prog_len; i++)
    locs[i] = yk_location_new();

  // The program counter.
  int pc = 0;

  NOOPT_VAL(pc);
  NOOPT_VAL(prog);
  NOOPT_VAL(prog_len);
  NOOPT_VAL(mem);
  NOOPT_VAL(locs);

  // interpreter loop.
  while (true) {
    YkLocation *loc = &locs[pc];
    yk_control_point(loc);
    assert(pc < prog_len);
    int bc = prog[pc];
    fprintf(stderr, "pc=%d, mem=%d\n", pc, mem);
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
  abort(); // FIXME: unreachable due to aborting guard failure earlier.
  NOOPT_VAL(pc);
  return (EXIT_SUCCESS);
}
