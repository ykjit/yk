// ## FIXME: PT IP filtering means we can't reliably detect longjmp() in
// ##        external code.
// ## FIXME: Implement setjmp/longjmp detection for swt.
// ignore-if: true
// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=3
//   stderr:
//     yk-tracing: start-tracing
//     set jump point
//     jumped!
//     yk-tracing: stop-tracing
//     yk-warning: trace-compilation-aborted: longjmp encountered
//     ...

// Check that we bork on a call to setjmp in unmapped code.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

void unmapped_setjmp(void);

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    unmapped_setjmp();
    i--;
  }

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
