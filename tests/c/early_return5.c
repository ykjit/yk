// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   stderr:
//     ...
//     yk-execution: return {"trid": "{{_}}"}
//     4 4 4

// Check that a compiled return side-trace preserves a constant return value
// across the call which notifies the runtime that the trace has returned.

#include <stdio.h>
#include <stdlib.h>
#include <yk.h>
#include <yk_testing.h>

static int loop(YkMT *mt, YkLocation *loc, int i) {
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, loc);
    if (i == 4)
      return 4;
    i--;
  }
  return 0;
}

int main(void) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 1);
  yk_mt_sidetrace_threshold_set(mt, 1);
  YkLocation loc = yk_location_new();

  NOOPT_VAL(loc);
  int x = loop(mt, &loc, 7);
  int y = loop(mt, &loc, 7);
  int z = loop(mt, &loc, 7);
  fprintf(stderr, "%d %d %d\n", x, y, z);

  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return EXIT_SUCCESS;
}
