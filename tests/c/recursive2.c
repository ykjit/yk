// Run-time:
//   env-var: YKD_LOG_IR=-:jit-pre-opt,jit-post-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YK_LOG=4
//   stderr:
//     yk-jit-event: start-tracing
//     0x{{loc2}}: 2
//     0x{{loc2}}: 1
//     yk-jit-event: stop-tracing-early-return
//     return
//     0x{{loc1}}: 3
//     yk-jit-event: start-tracing
//     0x{{loc1}}: 2
//     yk-jit-event: stop-tracing
//     --- Begin jit-pre-opt ---
//     ...
//     --- End jit-pre-opt ---
//     ...
//     0x{{loc1}}: 1
//     return
//     exit

// Check that early return from recursive interpreter loops works.
//
// In this scenario, the parent function starts tracing at location 1, a
// recursive interpreter loop runs and exits, but without encountering
// location 1 (the location that initiated tracing).
//
// XXX: question to Laurie: should the early_return from the inner interpreter
// loop abort tracing in this scenario? (It does, FWIW -- this test passes).

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int loop(YkMT *, YkLocation *, YkLocation *, int);

int loop(YkMT *mt, YkLocation *use_loc, YkLocation *next_loc, int i) {
  assert(use_loc != NULL);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, use_loc);
    if (i > 2) {
      loop(mt, next_loc, NULL, i - 1);
    }
    fprintf(stderr, "%p: %d\n", use_loc, i);
    i--;
  }
  yk_mt_early_return(mt);
  fprintf(stderr, "return\n");
  return i;
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);

  // First location: used by first level deep recursion.
  YkLocation loc1 = yk_location_new();
  // Second location: used by second level deep recursion.
  YkLocation loc2 = yk_location_new();

  NOOPT_VAL(loc1);
  NOOPT_VAL(loc2);
  loop(mt, &loc1, &loc2, 3);
  fprintf(stderr, "exit\n");
  yk_location_drop(loc1);
  yk_location_drop(loc2);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
