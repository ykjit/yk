// Compiler:
//   env-var: YKB_EXTRA_CC_FLAGS=-O2
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     yk-tracing: start-tracing
//     4: 0x{{faddr}}
//     yk-tracing: stop-tracing
//     3: 0x{{faddr}}
//     yk-execution: enter-jit-code {"trid": "0"}
//     2: 0x{{faddr}}
//     1: 0x{{faddr}}
//     yk-execution: deoptimise {"trid": "0", "gidx": "{{_}}"}
//     exit: 0x{{faddr}}

// Check that having a base frame pointer in a stackmap works correctly. On
// some platforms this probably doesn't test anything useful, but on others it
// does.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  void *outer = __builtin_frame_address(0);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    NOOPT_VAL(argc);
    fprintf(stderr, "%d: %p\n", i, outer);
    i--;
  }

  fprintf(stderr, "exit: %p", outer);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
