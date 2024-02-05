// Run-time:
//   env-var: YKD_PRINT_IR=jit-pre-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_PRINT_JITSTATE=1
//   stderr:
//     jit-state: start-tracing
//     pc=0, mem=4
//     pc=1, mem=4
//     pc=2, mem=4
//     pc=3, mem=3
//     jit-state: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     define ptr @__yk_compiled_trace_0(ptr %0, ptr %1, ptr %2) {
//       ...
//       %{{fptr}} = getelementptr %YkCtrlPointVars, ptr %0, i32 0, i32 0...
//       %{{load}} = load...
//       ...
//
//     {{bb}}:...
//     ...
//
//     {{another-bb}}:...
//       ...
//       %{{restart-cond}} = icmp sgt i32 %{{mem}}, 0...
//       br i1 %{{restart-cond}}, label %{{restart-bb}}, label %{{guard-fail-bb}}
//
//     {{guard-fail-bb}}:...
//       ...
//       %{{deoptret}} = call ptr (...) @llvm.experimental.deoptimize.p0(...
//       ret ptr %{{deoptret}}
//     ...
//
//     {{restart-bb}}:...
//       ...
//       br label...
//     }
//     ...
//     --- End jit-pre-opt ---
//     pc=0, mem=3
//     pc=1, mem=3
//     pc=2, mem=3
//     pc=3, mem=2
//     jit-state: enter-jit-code
//     pc=0, mem=2
//     pc=1, mem=2
//     pc=2, mem=2
//     pc=3, mem=1
//     pc=0, mem=1
//     pc=1, mem=1
//     pc=2, mem=1
//     pc=3, mem=0
//     jit-state: deoptimise
//     pc=4, mem=0
//     pc=5, mem=0

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

// The sole mutable memory cell of the interpreter.
int mem = 4;

// The bytecodes accepted by the interpreter.
#define NOP 0
#define DEC 1
#define RESTART_IF_NOT_ZERO 2
#define EXIT 3

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);

  // A hard-coded program to execute.
  int prog[] = {NOP, NOP, DEC, RESTART_IF_NOT_ZERO, NOP, EXIT};
  size_t prog_len = sizeof(prog) / sizeof(prog[0]);

  YkLocation loop_loc = yk_location_new();
  YkLocation **locs = malloc(6 * 8);
  for (int i = 0; i < prog_len; i++)
    if (i == 0)
      locs[i] = &loop_loc;
    else
      locs[i] = NULL;

  // The program counter.
  int pc = 0;

  NOOPT_VAL(pc);
  NOOPT_VAL(prog);
  NOOPT_VAL(prog_len);
  NOOPT_VAL(mem);
  NOOPT_VAL(locs);

  // interpreter loop.
  while (true) {
    assert(pc < prog_len);
    yk_mt_control_point(mt, locs[pc]);
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
  NOOPT_VAL(pc);

  yk_location_drop(loop_loc);
  yk_mt_drop(mt);

  return (EXIT_SUCCESS);
}
