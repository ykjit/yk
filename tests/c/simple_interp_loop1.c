// # Shadow stack doesn't currently support dynamically sized stack.
// ignore-if: true
// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_PRINT_JITSTATE=1
//   stderr:
//     jit-state: start-tracing
//     pc=0, mem=12
//     pc=1, mem=11
//     pc=2, mem=10
//     pc=3, mem=9
//     jit-state: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     define void @__yk_compiled_trace_0(%YkCtrlPointVars* %0, i64* %1, i64 %2) {
//       ...
//       %{{fptr}} = getelementptr %YkCtrlPointVars, %YkCtrlPointVars* %0, i32 0, i32 0...
//       %{{load}} = load...
//       ...
//       %{{cond}} = icmp...
//       ...
//       br...
//
//     {{guard-fail-bb}}:...
//       call void (...) @llvm.experimental.deoptimize.isVoid(i64* %1, i64 %2) ...
//       ret void
//
//     {{another-bb}}:...
//       ...
//       %{{fptr2}} = getelementptr %YkCtrlPointVars, %YkCtrlPointVars* %0, i32 0, i32 0...
//       store...
//       ...
//       ret void
//     }
//     ...
//     --- End jit-pre-opt ---
//     pc=0, mem=9
//     pc=1, mem=8
//     pc=2, mem=7
//     pc=3, mem=6
//     jit-state: enter-jit-code
//     pc=0, mem=6
//     pc=1, mem=5
//     pc=2, mem=4
//     pc=3, mem=3
//     jit-state: exit-jit-code
//     jit-state: enter-jit-code
//     pc=0, mem=3
//     pc=1, mem=2
//     pc=2, mem=1
//     pc=3, mem=0
//     jit-state: stopgap
//     ...
//     Indirect: 3 ...
//     ...

// Test basic interpreter.

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

// The sole mutable memory cell of the interpreter.
int mem = 12;

// The bytecodes accepted by the interpreter.
#define DEC 1
#define RESTART_IF_NOT_ZERO 2

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);

  // A hard-coded program to execute.
  int prog[] = {DEC, DEC, DEC, RESTART_IF_NOT_ZERO, DEC, DEC};
  size_t prog_len = sizeof(prog) / sizeof(prog[0]);

  // Create one location for each potential PC value.
  YkLocation loop_loc = yk_location_new();
  YkLocation *locs[prog_len];
  for (int i = 0; i < prog_len; i++)
    if (i == 0)
      locs[i] = &loop_loc;
    else
      locs[i] = NULL;

  // The program counter.
  int pc = 0;

  NOOPT_VAL(prog);
  NOOPT_VAL(prog_len);
  NOOPT_VAL(pc);
  NOOPT_VAL(mem);
  NOOPT_VAL(locs);

  // interpreter loop.
  while (true) {
    if (pc >= prog_len) {
      exit(0);
    }
    yk_mt_control_point(mt, locs[pc]);
    int bc = prog[pc];
    fprintf(stderr, "pc=%d, mem=%d\n", pc, mem);
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
  abort(); // FIXME: unreachable due to aborting guard failure earlier.
  NOOPT_VAL(pc);

  yk_location_drop(loop_loc);
  yk_mt_drop(mt);

  return (EXIT_SUCCESS);
}
