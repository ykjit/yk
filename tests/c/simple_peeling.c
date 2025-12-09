// ignore-if: test "$YK_JITC" = "j2" # not yet implemented in j2
// Run-time:
//   env-var: YKD_LOG_IR=jit-post-opt
//   env-var: YKD_SERIALISE_COMPILATION=1
//   env-var: YKD_LOG=3
//   stderr:
//     ...
//     --- Begin jit-post-opt ---
//     ...
//     header_start ...
//     ...
//     header_end [%{{0}}, %{{1}}, %{{2}}, %{{3}}]
//     ...
//     body_start [%{{19}}, %{{20}}, %{{21}}, %{{22}}]
//     ...
//     body_end ...
//     ...
//     --- End jit-post-opt ---
//     ...

// Check that basic trace peeling works.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yk.h>
#include <yk_testing.h>

int main(int argc, char **argv) {
  YkMT *mt = yk_mt_new(NULL);
  yk_mt_hot_threshold_set(mt, 0);
  YkLocation loc = yk_location_new();

  int res = 9998;
  int i = 4;
  NOOPT_VAL(loc);
  NOOPT_VAL(res);
  NOOPT_VAL(i);
  while (i > 0) {
    yk_mt_control_point(mt, &loc);
    fprintf(stderr, "%d\n", i);
    i--;
  }
  fprintf(stderr, "exit\n");
  NOOPT_VAL(res);
  yk_location_drop(loc);
  yk_mt_shutdown(mt);
  return (EXIT_SUCCESS);
}
