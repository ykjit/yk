// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     ...
//     yk-jit-event: start-tracing
//     i=0, loc0
//     i=1, loc1
//     yk-jit-event: stop-tracing (inner loop detected)
//     i=2, loc1
//     yk-jit-event: enter-jit-code
//     i=3, loc1
//     i=4, loc1
//     yk-jit-event: deoptimise
//     i=5, loc0
//     ...

// Check that we can switch to compiling an inner loop that we discover when
// tracing an outer loop.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

#define NUM_LOCS 3

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);

  YkLocation locs[NUM_LOCS];
  for (int i = 0; i < NUM_LOCS; i++) {
    locs[i] = yk_location_new();
  }

  // The locations we will encounter (in order) in the loop below.
  // We will start tracing at locs[0], but there is an inner loop starting
  // at locs[1], and it is the inner loop we should compile.
  YkLocation *loc_seq[] = {
    &locs[0], &locs[1], &locs[1], &locs[1], &locs[1], &locs[0], &locs[2] };

  int i = 0;
  NOOPT_VAL(i);
  for (; i < 7; i++) {
    YkLocation *loc = loc_seq[i];
    /* yk_debug_str("control point"); */
    yk_mt_control_point(mt, loc);

    char *s = NULL;
    if (loc == &locs[0]) {
      s = "loc0";
    } else if (loc == &locs[1]) {
      s = "loc1";
    } else if (loc == &locs[2]) {
      s = "loc2";
    } else {
      abort(); // unreachable
    }
    fprintf(stderr, "i=%d, %s\n", i, s);
  }

  for (int i = 0; i < NUM_LOCS; i++) {
    yk_location_drop(locs[i]);
  }
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
