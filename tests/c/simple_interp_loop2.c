// Run-time:
//   env-var: YK_LOG=4
//   env-var: YKD_LOG_STATS=/dev/null
//   stderr:
//     yk-jit-event: start-tracing
//     pc=0, mem=4
//     pc=1, mem=4
//     pc=2, mem=4
//     pc=3, mem=3
//     yk-jit-event: stop-tracing
//     pc=0, mem=3
//     pc=1, mem=3
//     pc=2, mem=3
//     pc=3, mem=2
//     yk-jit-event: enter-jit-code
//     pc=0, mem=2
//     pc=1, mem=2
//     pc=2, mem=2
//     pc=3, mem=1
//     pc=0, mem=1
//     pc=1, mem=1
//     pc=2, mem=1
//     pc=3, mem=0
//     yk-jit-event: deoptimise
//     pc=4, mem=0
//     pc=5, mem=0

// Test a basic interpreter.

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

bool test_compiled_event(YkCStats stats) {
  return stats.traces_compiled_ok == 1;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);

  // A hard-coded program to execute.
  int prog[] = {NOP, NOP, DEC, RESTART_IF_NOT_ZERO, NOP, EXIT};
  size_t prog_len = sizeof(prog) / sizeof(prog[0]);

  YkLocation loop_loc = yk_location_new();
  YkLocation **locs = calloc(prog_len, sizeof(&prog[0]));
  assert(locs != NULL);
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
    if ((pc == 0) && (mem == 3)) {
      __ykstats_wait_until(mt, test_compiled_event);
    }
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

  free(locs);
  yk_location_drop(loop_loc);
  yk_mt_shutdown(mt);

  return (EXIT_SUCCESS);
}
