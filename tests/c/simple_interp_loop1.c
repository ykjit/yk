// Run-time:
//   env-var: YKD_LOG=4
//   env-var: YKD_LOG_STATS=/dev/null
//   stderr:
//     yk-jit-event: start-tracing
//     pc=0, mem=12
//     pc=1, mem=11
//     pc=2, mem=10
//     pc=3, mem=9
//     yk-jit-event: stop-tracing
//     pc=0, mem=9
//     pc=1, mem=8
//     pc=2, mem=7
//     pc=3, mem=6
//     yk-jit-event: enter-jit-code
//     pc=0, mem=6
//     pc=1, mem=5
//     pc=2, mem=4
//     pc=3, mem=3
//     pc=0, mem=3
//     pc=1, mem=2
//     pc=2, mem=1
//     pc=3, mem=0
//     yk-jit-event: deoptimise
//     pc=4, mem=0
//     pc=5, mem=-1

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

bool test_compiled_event(YkCStats stats) {
  return stats.traces_compiled_ok == 1;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);

  // A hard-coded program to execute.
  int prog[] = {DEC, DEC, DEC, RESTART_IF_NOT_ZERO, DEC, DEC};
  size_t prog_len = sizeof(prog) / sizeof(prog[0]);

  // Create one location for each potential PC value.
  YkLocation *locs = calloc(prog_len, sizeof(&prog[0]));
  for (int i = 0; i < prog_len; i++)
    if (i == 0)
      locs[i] = yk_location_new();
    else
      locs[i] = yk_location_null();

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
    yk_mt_control_point(mt, &locs[pc]);
    if ((pc == 0) && (mem == 9)) {
      __ykstats_wait_until(mt, test_compiled_event);
    }
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

  free(locs);
  yk_mt_shutdown(mt);

  return (EXIT_SUCCESS);
}
