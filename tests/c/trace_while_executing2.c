// Run-time:
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=4
//   env-var: YKD_LOG_IR=jit-pre-opt
//   stderr:
//     yk-tracing: start-tracing
//     0: 5
//     yk-tracing: stop-tracing
//     --- Begin jit-pre-opt ---
//       ...
//       call %{{_}}(%{{_}}, %{{_}}, %{{_}}, %{{_}}, %{{_}}) ; @__yk_opt_indirect
//       ...
//     --- End jit-pre-opt ---
//     0: 4
//     yk-execution: enter-jit-code
//     0: 3
//     yk-execution: deoptimise ...
//     yk-tracing: start-tracing
//     1: 3
//     yk-tracing: stop-tracing
//     ...
//     0: 2
//     yk-execution: enter-jit-code
//     0: 1
//     yk-execution: deoptimise...
//     exit

// Used to test that we don't record a successful trace if we started tracing
// in an inner control point. Now checks that we do generate a valid trace from
// the inner control point.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

void indirect(YkMT *, YkLocation *, YkLocation *, int, int);

void loop(YkMT *mt, YkLocation *loc1, YkLocation *loc2, int i, int depth) {
  YkLocation *loc = loc1;
  if (depth == 1)
    loc = loc2;
  while (i > 0) {
    yk_mt_control_point(mt, loc);
    fprintf(stderr, "%d: %d\n", depth, i);
    if (i == 3) {
      if (depth == 0)
        loc = loc2;
      else
        return;
    }
    indirect(mt, loc1, loc2, i, depth);
    i--;
  }
}

__attribute__((yk_outline))
void indirect(YkMT *mt, YkLocation *loc1, YkLocation *loc2, int i, int depth) {
  if (depth == 0 && i == 3)
    loop(mt, loc1, loc2, i, 1);
}

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc1 = yk_location_new();
  YkLocation loc2 = yk_location_new();

  loop(mt, &loc1, &loc2, 5, 0);
  yk_location_drop(loc1);
  yk_location_drop(loc2);
  yk_mt_shutdown(mt);
  fprintf(stderr, "exit\n");
  return (EXIT_SUCCESS);
}
